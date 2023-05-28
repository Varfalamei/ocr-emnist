from dataclasses import dataclass
from typing import Tuple, Optional

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
    num_classes: Optional[int]
    blank_label: Optional[int] = None
