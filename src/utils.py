import sys
from itertools import groupby

import Levenshtein
import numpy as np
import torch
import torchvision
from colorama import Fore
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random

from src.configs import ExpCONFIG
from src.dataset import CapchaDataset


def train_epoch(model, criterion, optimizer, data_loader, config: ExpCONFIG) -> None:
    model.train()
    train_correct = 0
    train_total = 0
    total_distance = 0
    for x_train, y_train in tqdm(
        data_loader,
        position=0,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    ):
        batch_size = x_train.shape[0]
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        optimizer.zero_grad()
        y_pred = model(x_train.to(config.device))
        y_pred = y_pred.permute(1, 0, 2)
        input_lengths = torch.IntTensor(batch_size).fill_(model.cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        _, max_index = torch.max(
            y_pred, dim=2
        )
        for i in range(batch_size):
            raw_prediction = list(
                max_index[:, i].detach().cpu().numpy()
            )
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != config.blank_label]
            )
            prediction = prediction.to(torch.int32)[prediction != config.blank_label]
            y_true = y_train[i].to(torch.int32)[y_train[i] != config.blank_label]

            if len(prediction) == len(y_true) and torch.all(prediction.eq(y_true)):
                train_correct += 1
            train_total += 1
            total_distance += Levenshtein.distance(prediction.tolist(), y_true.tolist())
    acc = train_correct / train_total
    avg_distance = total_distance / train_total
    print('TRAINING. Correct (accuracy): ', train_correct, '/', train_total, '=', acc)
    print("TRAINING. Average Levenshtein distance:", avg_distance)


def valid_epoch(model, criterion, val_loader, config: ExpCONFIG) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        total_distance = 0
        for x_val, y_val in tqdm(
            val_loader,
            position=0,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
        ):
            batch_size = x_val.shape[0]
            x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
            y_pred = model(x_val.to(config.device))
            y_pred = y_pred.permute(1, 0, 2)
            input_lengths = torch.IntTensor(batch_size).fill_(model.cnn_output_width)
            target_lengths = torch.IntTensor([len(t) for t in y_val])
            criterion(y_pred, y_val, input_lengths, target_lengths)
            _, max_index = torch.max(y_pred, dim=2)
            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                prediction = torch.IntTensor(
                    [c for c, _ in groupby(raw_prediction) if c != config.blank_label]
                )
                prediction = prediction.to(torch.int32)[prediction != config.blank_label]
                y_true = y_val[i].to(torch.int32)[y_val[i] != config.blank_label]

                if len(prediction) == len(y_true) and torch.all(prediction.eq(y_true)):
                    val_correct += 1
                val_total += 1
                total_distance += Levenshtein.distance(prediction.tolist(), y_true.tolist())
        acc = val_correct / val_total
        avg_distance = total_distance / val_total
        print("VALID. Correct (accuracy): ", val_correct, "/", val_total, "=", acc)
        print("VALID. Average Levenshtein distance:", avg_distance)
    return acc, avg_distance


def test_model(model, test_loader, config: ExpCONFIG):
    model.eval()
    test_preds = []
    (x_test, y_test) = next(iter(test_loader))
    y_pred = model(
        x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).to(config.device)
    )
    y_pred = y_pred.permute(1, 0, 2)
    _, max_index = torch.max(y_pred, dim=2)
    for i in range(x_test.shape[0]):
        raw_prediction = list(max_index[:, i].detach().cpu().numpy())
        prediction = torch.IntTensor(
            [c for c, _ in groupby(raw_prediction) if c != config.blank_label]
        )
        test_preds.append(prediction)

    for j in range(len(x_test)):
        mpl.rcParams["font.size"] = 8
        plt.imshow(x_test[j], cmap="gray")
        mpl.rcParams["font.size"] = 18
        plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(y_test[j].numpy()))
        plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(test_preds[j].numpy()))
        plt.savefig(f"{config.logdir}/{config.arch}/plot_{j}.png")
        plt.show()


def generate_captcha_image_from_emnist(CONFIG):
    dataset = CapchaDataset((6, 7))
    set_of_captcha = dataset[0]
    image = set_of_captcha[0]
    captcha_text = set_of_captcha[1]
    captcha_text = ''.join([str(int(digit)) if digit != 10 else '' for digit in captcha_text])
    plt.imshow(image, cmap='gray')
    plt.show()
    print(f"Generated Captcha with text {captcha_text}")
    return image


def generate_captcha_image(CONFIG):
    image = Image.new("RGB", (140, 28), "black")
    draw = ImageDraw.Draw(image)
    # Загружаем шрифт для отображения текста на изображении
    font = ImageFont.truetype("./src/fonts/Cookiesandcream.ttf", size=30)
    # Генерируем случайное количество цифр для капчи (от 3 до 5)
    num_digits = random.randint(CONFIG.len_of_mnist_sequence[0], CONFIG.len_of_mnist_sequence[1])
    # Генерируем случайный текст капчи (только цифры)
    captcha_text = "  ".join(random.choices("1234567890", k=num_digits))
    # Размещаем текст посередине изображения
    text_width, text_height = draw.textsize(captcha_text, font=font)
    x = (140 - text_width) // 2
    y = (28 - text_height) // 2
    draw.text((x, y), captcha_text, font=font, fill="white")
    image = image.convert("L")
    plt.imshow(image, cmap='gray')
    plt.show()
    print(f"Generated Captcha: {image} with text {captcha_text.replace(' ', '')}")
    return image
