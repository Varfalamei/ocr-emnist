import os
import numpy as np
from torch.utils.data import Dataset

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from pathlib import Path

from src.configs import ExpCONFIG
from src.dataset import CapchaDataset
from src.models import CRNN_v2
from src.utils import valid_epoch, train_epoch

CONFIG = ExpCONFIG(
    seed=42,
    epochs=5,
    batch_size=32,
    arch="cnn-gru-ctc-v2-ocr-system",
    logdir="./checkpoints",
    init_lr=0.001,
    validation_split=0.2,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    len_of_mnist_sequence=(3, 5),
    digits_per_sequence=5,
)


def train():
    dataset = CapchaDataset(CONFIG.len_of_mnist_sequence)
    CONFIG.num_classes = dataset.num_classes
    CONFIG.blank_label = dataset.blank_label

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(CONFIG.validation_split * dataset_size))
    np.random.seed(CONFIG.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG.batch_size, sampler=train_sampler, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG.batch_size, sampler=valid_sampler, drop_last=False
    )

    model = CRNN_v2(CONFIG.num_classes).to(CONFIG.device)

    criterion = nn.CTCLoss(
        blank=CONFIG.blank_label, reduction="mean", zero_infinity=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.init_lr)

    current_acc = 0
    for epoch in range(1, CONFIG.epochs + 1):
        print(f"Epoch: {epoch}/{CONFIG.epochs}")
        train_epoch(model, criterion, optimizer, train_loader, config=CONFIG)
        acc, _ = valid_epoch(model, val_loader, config=CONFIG)
        if acc > current_acc:
            if not os.path.exists(CONFIG.logdir):
                Path(CONFIG.logdir).mkdir()
            if not os.path.exists(CONFIG.logdir + "/" + CONFIG.arch):
                Path(CONFIG.logdir + "/" + CONFIG.arch).mkdir()

            model_out_name = (
                CONFIG.logdir
                + "/"
                + CONFIG.arch
                + f"/checkpoint_{epoch}_epoch_{round(acc * 100)}_acc.pt"
            )

            torch.save(model.state_dict(), model_out_name)


def generate_answers():
    pass


if __name__ == "__main__":
    # train()
    generate_answers()
