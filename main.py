import torch
import torchvision

from src.configs import ExpCONFIG
from src.models import CRNN_v2
from src.utils import generate_captcha_image

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


def load_model():
    model = CRNN_v2(CONFIG.num_classes).to(CONFIG.device)
    checkpoint = torch.load("checkpoints/cnn-gru-ctc-v2-ocr-system/checkpoint_5_epoch_95_acc.pt")
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
        captcha_text += str(label.item())

    return captcha_text


if __name__ == "__main__":
    model = load_model()
    captcha = generate_captcha_image(CONFIG=CONFIG)
    recognized_text = recognize_captcha(captcha, model)
    print("Generated Captcha:", captcha)
    print("Recognized Text:", recognized_text)
