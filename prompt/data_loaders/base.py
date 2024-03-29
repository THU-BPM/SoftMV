from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict
from prompt.utils import InputExample

import log

logger = log.get_logger("root")


class InfoEgs(object):
    def __init__(self, egs: List[InputExample], split_: str, lang: str):
        self.egs = egs
        self.split_ = split_
        self.lang = lang

    def __getitem__(self, idx):
        return self.egs[idx]

    def __len__(self):
        return len(self.egs)


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)
        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        for label in self._labels:
            if (
                self._examples_per_label[label] < self._max_examples[label]
                or self._max_examples[label] < 0
            ):
                return False
        return True

    def add(self, example: InputExample) -> bool:
        label = example.label
        if (
            self._examples_per_label[label] < self._max_examples[label]
            or self._max_examples[label] < 0
        ):
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        pass
