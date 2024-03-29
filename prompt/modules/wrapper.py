import json
import jsonpickle
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .configs import (
    WrapperConfig,
    MODEL_CLASSES,
    PREPROCESSORS,
    InputExample,
    PLM_WRAPPER,
    MLM_WRAPPER,
    get_linear_schedule_with_warmup,
    TRAIN_STEP_FUNCTIONS,
    EVALUATION_STEP_FUNCTIONS,
    CONFIG_NAME,
)
from prompt.utils import InputFeatures, DictDataset
from prompt.trainers.singleton_evalfns import singleton_evaluate
from prompt.preprocessor import Preprocessor
from tqdm import tqdm
import optuna
import log

logger = log.get_logger("root")


class ContinuousPrompt(nn.Module):
    def __init__(self, config: WrapperConfig, tokenizer, random_range=0.5):
        super(ContinuousPrompt, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_size = config.embed_size
        self.hidden_size = self.embed_size
        self.prompt_length = config.prompt_length
        self.random_range = random_range

        config_class = MODEL_CLASSES[self.config.model_type]["config"]
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False,
        )

        model_class = MODEL_CLASSES[self.config.model_type][MLM_WRAPPER]
        self.model = model_class.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )

        if config.soft_prompt_path is not None:
            self.set_soft_prompt_embeds()
        elif self.prompt_length > 0:
            self.initialize_soft_prompt()

        if config.prompt_encoder_type == "lstm":
            self.lstm_head = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.mlp_head = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif config.prompt_encoder_type == "mlp":
            self.mlp = nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
            )

        
    def set_soft_prompt_embeds(self) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path

        """
        self.prompt_embeddings = torch.load(
            self.config.soft_prompt_path, map_location=torch.device("cpu")
        )
        self.prompt_length = self.prompt_embeddings.num_embeddings
        logger.info(f"Set soft prompt! (n_tokens: {self.prompt_length})")

    def initialize_soft_prompt(self):
        if self.config.init_from_vocab:
            init_prompt_value = self.model.get_input_embeddings().weight[:self.prompt_length].clone().detach()
            print("init_prompt_value vocab", init_prompt_value.shape)
        else:
            init_prompt_value = torch.FloatTensor(self.prompt_length, self.embed_size).uniform_(
                -self.random_range, self.random_range
            )
            print("init_prompt_value random", init_prompt_value.shape)
        self.prompt_embeddings = nn.Embedding(self.prompt_length, self.embed_size)
        # Initialize weight
        self.prompt_embeddings.weight = nn.parameter.Parameter(init_prompt_value)

    def forward(
        self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )

class TransformerModelWrapper:
    def __init__(self, config: WrapperConfig, train_config: WrapperConfig):
        self.config = config
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kld_criterion = nn.KLDivLoss(reduction='batchmean')

        tokenizer_class = MODEL_CLASSES[self.config.model_type]["tokenizer"]
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )  
        # type: PreTrainedTokenizer

        self.model = ContinuousPrompt(config, self.tokenizer)

        self.preprocessor = PREPROCESSORS[MLM_WRAPPER](
            self,
            self.config.task_name,
            pattern_lang=train_config.pattern_lang,
            pattern_id=self.config.pattern_id,
        )


    @classmethod
    def from_pretrained(cls, path: str) -> "TransformerModelWrapper":
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]["tokenizer"]
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer)
        model_class = MODEL_CLASSES[wrapper.config.model_type][MLM_WRAPPER]
        wrapper.model.model = model_class.from_pretrained(path)

        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)
        wrapper.model.prompt_embeddings.load_state_dict(data["prompt_embeddings"])
        if "lstm_head" in data:
            assert "mlp_head" in data
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        wrapper.preprocessor = PREPROCESSORS[MLM_WRAPPER](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id
        )

        return wrapper

    def save(self, path: str) -> None:
        logger.info("Saving models")
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)
        if self.config.prompt_encoder_type == "lstm":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "lstm_head": model_to_save.lstm_head.state_dict(),
                "mlp_head": model_to_save.mlp_head.state_dict(),
            }
        elif self.config.prompt_encoder_type == "mlp":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "mlp": model_to_save.mlp.state_dict(),
            }
        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), "r") as f:
            return jsonpickle.decode(f.read())

    def eval(self, eval_dataloader, device, zs_infer=False) -> Dict:
        self.model.eval()
        preds, all_indices, out_label_ids = None, None, None
        all_guids = []
        for batch in tqdm(eval_dataloader):
            guids = batch["guids"]
            all_guids.extend(guids)
            if not zs_infer:
                batch = {k: t.to(device) for k, t in batch.items() if k != "guids"}
            else:
                batch = {k: t.cuda() for k, t in batch.items() if k != "guids"}
            labels = batch["labels"]
            indices = batch["idx"]
            with torch.no_grad():
                logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(
                    batch
                )
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0
                )
        self.model.train()
        return {
            "indices": all_indices,
            "logits": preds,
            "labels": out_label_ids,
            "guids": all_guids,
        }

    def generate_default_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        bz = batch["input_ids"].shape[0]
        block_flag = batch["block_flag"]
        model = self.model.module if hasattr(self.model, "module") else self.model
        if self.config.model_type == "albert":
            raw_embeds = model.model.albert.embeddings.word_embeddings(input_ids)
        elif self.config.model_type == "bert":
            raw_embeds = model.model.bert.embeddings.word_embeddings(input_ids)
        elif self.config.model_type in ["roberta", "xlm-roberta"]:
            raw_embeds = model.model.roberta.embeddings.word_embeddings(input_ids)
        else:
            raise ValueError("Unsupported model_type")
        
        replace_embeds = model.prompt_embeddings.weight.unsqueeze(0)
        # replace_embeds = model.prompt_embeddings(
        #     torch.LongTensor(list(range(model.prompt_length))).to(self.rank_idx)
        # )
        # replace_embeds = replace_embeds.unsqueeze(0)
        if self.config.prompt_encoder_type == "lstm":
            replace_embeds = model.lstm_head(replace_embeds)[0]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head(replace_embeds)
            else:
                replace_embeds = model.mlp_head(replace_embeds).squeeze()
        elif self.config.prompt_encoder_type == "mlp":
            replace_embeds = model.mlp(replace_embeds)
        blocked_indices = (
            (block_flag == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]
        )
        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        inputs = {
            "inputs_embeds": raw_embeds,
            "attention_mask": batch["attention_mask"],
        }

        if self.config.model_type in ["bert"]:
            inputs["token_type_ids"] = batch["token_type_ids"]

        return inputs

    def mlm_train_step(
        self,
        labeled_batch1: Dict[str, torch.Tensor],
        labeled_batch2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        inputs1 = self.generate_default_inputs(labeled_batch1)
        inputs2 = self.generate_default_inputs(labeled_batch2)
        mlm_labels1, labels = labeled_batch1["mlm_labels"], labeled_batch1["labels"]
        mlm_labels2 = labeled_batch2["mlm_labels"]

        outputs1 = self.model(**inputs1)
        prediction_scores1 = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits_uni(
            mlm_labels1, outputs1[0]
        )
        # prediction_scores1 = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
        #     mlm_labels1, outputs1[0]
        # )
        outputs2 = self.model(**inputs2)

        # prediction_scores2 = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
        #     mlm_labels2, outputs2[0]
        # )
        prediction_scores2 = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits_uni(
            mlm_labels2, outputs2[0]
        )
        loss1 = self.ce_criterion(
            prediction_scores1.view(-1, len(self.config.label_list)), labels.view(-1)
        )
        loss2 = self.ce_criterion(
            prediction_scores2.view(-1, len(self.config.label_list)), labels.view(-1)
        )
        loss3 = self.kld_criterion(F.log_softmax(prediction_scores1, dim=-1), F.softmax(prediction_scores2, dim=-1))
        return  loss1 + loss2 + 0.3 * loss3
        # return loss1 + loss2 + loss3

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits_uni(
            batch["mlm_labels"], outputs[0]
        )
        # return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
        #     batch["mlm_labels"], outputs[0]
        # )

    def train(
        self,
        # trial,
        rank_idx,
        data_iterator,
        eval_config,
        device,
        n_gpu: int = 1,
        num_train_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        warmup_steps=0,
        max_grad_norm: float = 1,
        logging_steps: int = 50,
        use_logits: bool = False,
        ddp_config=None,
        **_,
    ):
        self.ce_criterion = self.ce_criterion.to(device)
        train_batch_size = data_iterator.trn_batch_size
        train_dataloader = data_iterator.trn_iter
        train_dataloader_aug = data_iterator.trn_iter_aug
        val_dataloader = data_iterator.val_iter
        t_total = (
            len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # Generate optimizer
        # lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate
        )


        if n_gpu > 1 and ddp_config is None:
            self.model = torch.nn.DataParallel(self.model)
        step, global_step = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        self.model.train()
        best_score, best_model = -1, None
        for epoch_index in range(int(num_train_epochs)):
            if ddp_config:
                data_iterator.train_sampler.set_epoch(epoch_index)
            for _, (batch1,batch2) in enumerate(zip(train_dataloader,train_dataloader_aug)): 
                batch1 = {k: t.to(device) for k, t in batch1.items() if k != "guids"}
                batch2 = {k: t.to(device) for k, t in batch2.items() if k != "guids"}
                loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch1, batch2)
                if n_gpu > 1 and not ddp_config:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = learning_rate
                        logs["learning_rate"] = learning_rate_scalar
                        logs["regular_trn_loss"] = loss_scalar
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{"step": global_step}}))
                if step % 4 == 0:
                    logger.info(
                        f"rank{rank_idx} loss: {loss.item():.4f}, batch: {step}, epoch: {epoch_index}"
                    )
                step += 1
            logger.info(
                f"rank {rank_idx} epoch {epoch_index} finished. start evaluating."
            )
            eval_results = self.eval(val_dataloader, device)
            eval_results, _ = singleton_evaluate(eval_results, eval_config)
            logger.info(f"{eval_results['scores']}")
            if eval_results["scores"]["acc"] > best_score:
                best_score = eval_results["scores"]["acc"]
                best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
        if rank_idx == 0:
            return (
                global_step,
                (tr_loss / global_step if global_step > 0 else -1),
                best_score,
                best_model,
            )
        else:
            return (None, None, None, None)
