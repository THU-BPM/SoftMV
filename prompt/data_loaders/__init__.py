from .xnli_dataset import XNLIDataset

name2datasets = {
    "xnli": XNLIDataset,
}  
# type: Dict[str,Callable[[],DataProcessor]]


METRICS = {
    "xnli": ["acc"],
}

DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]


