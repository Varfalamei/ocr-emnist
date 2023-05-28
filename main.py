import torch
import torchvision

from src.configs import ExpCONFIG
from src.models import CRNN_v2, CRNN_v1
from src.utils import generate_captcha_image, generate_captcha_image_from_emnist

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
    num_classes=11,
)


def load_model():
    model = CRNN_v2(CONFIG.num_classes).to(CONFIG.device)
    checkpoint = torch.load(
        "checkpoints/cnn-gru-ctc-v2-ocr-system/checkpoint_5_epoch_95_acc.pt",
        map_location=CONFIG.device
    )
    model.load_state_dict(checkpoint)
    return model


def recognize_captcha(image, model):
    # Преобразование изображения в тензор и нормализация
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(CONFIG.device)

    # Распознавание текста капчи с помощью модели OCR
    with torch.no_grad():
        predictions = model(image_tensor)
        _, predicted_labels = torch.max(predictions, dim=2)

    # Преобразование предсказанных меток в текст
    captcha_text = ""
    for label in predicted_labels[0]:
        if label != 10:
            captcha_text += str(label.item())

    print("Recognized Text:", captcha_text)


if __name__ == "__main__":
    model = load_model()
    captcha = generate_captcha_image_from_emnist()
    recognize_captcha(captcha, model)
