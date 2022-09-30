from abc import ABC
from typing import List
import json


class promptConfig(ABC):
    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        cfg = cls.__new__(cls)
        with open(path, "r", encoding="utf8") as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(promptConfig):
    def __init__(
        self,
        pattern_lang: str,
        device: str = None,
        per_gpu_train_batch_size: int = 8,
        n_gpu: int = 1,
        num_train_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        max_grad_norm: float = 1,
        use_logits: bool = False,
        seed: int = 1,
        cosda_rate: float = 0.2,
    ):
        self.pattern_lang = pattern_lang
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.use_logits = use_logits
        self.seed = seed
        self.cosda_rate = cosda_rate


class EvalConfig(promptConfig):
    def __init__(
        self,
        device: str = None,
        n_gpu: int = 1,
        per_gpu_eval_batch_size: int = 8,
        metrics: List[str] = None,
    ):
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics


class DDPConfig(promptConfig):
    def __init__(self, do_ddp, num_ranks, num_nodes, world_size=-1):
        super().__init__()
        self.do_ddp = do_ddp
        self.num_ranks = num_ranks
        self.num_nodes = num_nodes
        self.world_size = world_size


class IpromptConfig(promptConfig):
    pass
