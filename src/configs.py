from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ExpCONFIG:
    seed: int
    epochs: int
    batch_size: int
    arch: str
    logdir: str
    init_lr: float
    validation_split: float
    device: torch.device
    len_of_mnist_sequence: Tuple[int, int]
    digits_per_sequence: int
    num_classes: int = None
    blank_label: int = None
