from secrets import token_bytes
from pet.utils import InputFeatures, DictDataset, InputExample
from pet.modules.configs import PLM_WRAPPER
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from typing import List, Dict
import torch
import numpy as np
import log
import os
import random
from copy import copy, deepcopy

logger = log.get_logger("root")


class PatternizedIterator(object):
    def __init__(
        self,
        dataset,
        preprocessor,
        pattern_lang: str,
        trn_batch_size: int,
        inf_batch_size: int,
        dict_dir: str,
        cosda_rate: float,
    ):
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.pattern_lang = pattern_lang
        self.trn_batch_size = trn_batch_size
        self.inf_batch_size = inf_batch_size
        self.trn_iter = None
        self.val_iter = None
        self.zs_iters = None
        self.trn_iter_aug = None
        self.dict_dir = dict_dir
        self.languages = ['zh', 'de', 'fr', 'ar', 'bg', 'el', 'es', 'hi', 'ru', 'th', 'tr', 'vi']
        if self.dict_dir is not None:      
            self.word_dict = self.get_word_dict(self.dict_dir, self.languages)
        self.cosda_rate = cosda_rate
        self.cosda_lang = []

    def patternize_trn(self):
        logger.info(f"Start patternizing training data (trn_egs)")
        logger.info(f"Traning data language: {self.dataset.trn_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features = self._generate_dataset(self.dataset.trn_egs.egs)
        self.trn_iter = self._wrap_sampler("trn", features)

    def patternize_val(self):
        logger.info(f"Start patternizing validation data (val_egs)")
        logger.info(f"Val data language: {self.dataset.val_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features = self._generate_dataset(self.dataset.val_egs.egs)
        self.val_iter = self._wrap_sampler("val", features)

    def patternize_zs(self):
        logger.info(f"Start patternizing zs transfer data (zs_egs)")
        logger.info(f"Pattern language: {self.pattern_lang}")
        self.zs_iters = {lang: None for lang, _ in self.dataset.zs_egs.items()}
        for lang, egs in self.dataset.zs_egs.items():
            logger.info(f"ZS data language: {egs.lang}")
            features = self._generate_dataset(egs)
            self.zs_iters[lang] = self._wrap_sampler("zs", features)

    def pct_patternize_trn(self, output_dir):
        logger.info(f"Start patternizing training data (trn_egs)")
        logger.info(f"Traning data language: {self.dataset.trn_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features, features_target = self._pct_generate_dataset(self.dataset.trn_egs.egs, output_dir)
        self.trn_iter = self._pct_wrap_sampler("trn", features)
        self.trn_iter_aug = self._pct_wrap_sampler("trn", features_target)

    def ddp_patternize_trn(self, ddp_config, rank_idx):
        logger.info(f"Refactoring trn_egs with DDP wrapper")
        logger.info(f"Traning data language: {self.dataset.trn_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features = self._generate_dataset(self.dataset.trn_egs.egs)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            features,
            num_replicas=ddp_config.world_size,
            rank=rank_idx,
            shuffle=True,
        )
        self.trn_iter = torch.utils.data.DataLoader(
            dataset=features,
            batch_size=self.trn_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=self.train_sampler,
        )

    def _wrap_sampler(self, split_, features):
        sampler = RandomSampler if split_ == "trn" else SequentialSampler
        bs = self.trn_batch_size if split_ == "trn" else self.inf_batch_size
        return DataLoader(features, sampler=sampler(features), batch_size=bs)

    def _pct_wrap_sampler(self, split_, features):
        sampler = SequentialSampler
        bs = self.trn_batch_size if split_ == "trn" else self.inf_batch_size
        return DataLoader(features, sampler=sampler(features), batch_size=bs)

    def _convert_examples_to_features(
        self, examples: List[InputExample]
    ) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.preprocessor.get_input_features(example)
            features.append(input_features)
            # if ex_index < 1:
            #     logger.info(f"--- Example {ex_index} ---")
            #     logger.info(
            #         input_features.pretty_print(self.preprocessor.wrapper.tokenizer)
            #     )
        return features

    def _pct_convert_examples_to_features(
        self, examples: List[InputExample], output_dir
    ) -> List[InputFeatures]:
        features = []
        features_target = []
        languages = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.preprocessor.get_input_features(example)
            if self.cosda_rate > 0:
                aug_example = self.positive_sample(example)
            else:
                aug_example = example
            input_features_target, language_sample = self.preprocessor.pct_get_input_features(aug_example)
            features.append(input_features)
            features_target.append(input_features_target)
            languages.append(language_sample)
            if ex_index < 1:
                logger.info(f"--- Example {ex_index} ---")
                logger.info(
                    input_features.pretty_print(self.preprocessor.wrapper.tokenizer)
                )
                logger.info(
                    input_features_target.pretty_print(self.preprocessor.wrapper.tokenizer)
                )
            #     logger.info(languages)
        with open(
            os.path.join(output_dir, "language.txt"), "w"
        ) as fh:
            fh.write("\n".join(self.cosda_lang))
        return features, features_target

    def _generate_dataset(self, data: List[InputExample]):
        features = self._convert_examples_to_features(data)
        feature_dict = {
            "input_ids": torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [f.token_type_ids for f in features], dtype=torch.long
            ),
            "labels": torch.tensor([f.label for f in features], dtype=torch.long),
            "mlm_labels": torch.tensor(
                [f.mlm_labels for f in features], dtype=torch.long
            ),
            "logits": torch.tensor([f.logits for f in features], dtype=torch.float),
            "idx": torch.tensor([f.idx for f in features], dtype=torch.long),
            "block_flag": torch.tensor(
                [f.block_flag for f in features], dtype=torch.long
            ),
            "guids": np.array([f.guid for f in features]),
        }
        return DictDataset(**feature_dict)

    def _pct_generate_dataset(self, data: List[InputExample], output_dir):
        features1, features2 = self._pct_convert_examples_to_features(data, output_dir)
        feature_dict1 = {
            "input_ids": torch.tensor(
                [f.input_ids for f in features1], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f.attention_mask for f in features1], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [f.token_type_ids for f in features1], dtype=torch.long
            ),
            "labels": torch.tensor([f.label for f in features1], dtype=torch.long),
            "mlm_labels": torch.tensor(
                [f.mlm_labels for f in features1], dtype=torch.long
            ),
            "logits": torch.tensor([f.logits for f in features1], dtype=torch.float),
            "idx": torch.tensor([f.idx for f in features1], dtype=torch.long),
            "block_flag": torch.tensor(
                [f.block_flag for f in features1], dtype=torch.long
            ),
            "guids": np.array([f.guid for f in features1]),
        }
        feature_dict2 = {
            "input_ids": torch.tensor(
                [f.input_ids for f in features2], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f.attention_mask for f in features2], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [f.token_type_ids for f in features2], dtype=torch.long
            ),
            "labels": torch.tensor([f.label for f in features2], dtype=torch.long),
            "mlm_labels": torch.tensor(
                [f.mlm_labels for f in features2], dtype=torch.long
            ),
            "logits": torch.tensor([f.logits for f in features2], dtype=torch.float),
            "idx": torch.tensor([f.idx for f in features2], dtype=torch.long),
            "block_flag": torch.tensor(
                [f.block_flag for f in features2], dtype=torch.long
            ),
            "guids": np.array([f.guid for f in features2]),
        }
        return DictDataset(**feature_dict1), DictDataset(**feature_dict2)

    # get positive sample
    def positive_sample(self,example):
        assert (self.word_dict is not None) and (self.languages is not None) and (self.cosda_rate is not None)
        aug_example = deepcopy(example)
        token_a_list = aug_example.text_a.strip().split()
        token_b_list = aug_example.text_b.strip().split()
        while "" in token_a_list:
            token_a_list.remove("")
        while "" in token_b_list:
            token_b_list.remove("")
        aug_example.text_a = " ".join(self.convert_token_with_CoSDA(
            tokens=token_a_list,
            word_dicts=self.word_dict,
            langs=self.languages,
            random_rate=self.cosda_rate))
        aug_example.text_b = " ".join(self.convert_token_with_CoSDA(
            tokens=token_b_list,
            word_dicts=self.word_dict,
            langs=self.languages,
            random_rate=self.cosda_rate))
        return aug_example

    def get_word_dict(self, word_dict_dir, langs: list):
        word_dicts = {lang: {} for lang in langs}
        for lang in langs:
            with open(word_dict_dir + lang + ".txt", "r",encoding='utf-8') as f:
                for line in f.readlines():
                    if not line:
                        continue
                    source, target = line.split()
                    if source.strip() == target.strip():
                        continue
                    if source not in word_dicts[lang]:
                        word_dicts[lang][source] = [target]
                    else:
                        word_dicts[lang][source].append(target)
        return word_dicts

    def convert_4_1_token(self, token, word_dicts, langs):
        this_lang = random.choice(langs)
        time = 0
        while time < 10 and token not in word_dicts[this_lang]:
            this_lang = random.choice(langs)
            time += 1

        if token in word_dicts[this_lang]:
            token = random.choice(word_dicts[this_lang][token])
            self.cosda_lang.append(this_lang)
            return token
        else:
            return token

    # code switch
    def convert_token_with_CoSDA(self,tokens, word_dicts, langs, random_rate=0.2):
        result = []
        for token in tokens:
            raw_token = token.replace("▁", "")
            if random.random() <= random_rate:
                result.append(self.convert_4_1_token(raw_token, word_dicts, langs))
            else:
                result.append(token)
        return deepcopy(result)