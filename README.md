# OCR system for recognition handwritten digits from any captchas

Домашнее задание к [задаче](https://github.com/aitalents/computer-vision-technology/tree/main/Topic%204.%20OCR)
и сделано оно по [этому](https://github.com/Alek-dr/OCR-Example/tree/master) репозиторию.
Также была реализована архитектуры из этой [статьи](https://arxiv.org/abs/1507.05717), эта модель называется CRNN_v2


## Пункты

- [x] архитектура реализована точно как описано в статье, а не просто похожа
- [x] выбрана и обоснована метрика качества
- [x] модель обучена и запускается в демо режиме - т.е. есть скрипт, который загружает обученную модель, на лету генерируется капча, модель её распознает
- [x] есть понимание работы CTC loss
- [ ] ответы на вопросы
- [ ] студент сдал и защитил работу в кратчайший срок, на ближайшей официальной практике после лекции

## Метрика качества
+ Accuracy - отражает насколько точно мы распознаем капчу целиком;
+ Levenshtein distance - отображает насколько хорошо побуквенно модель распознает капчу.

| Model             | Accuracy | Levenshtein distance    |
|-------------------|----------|--------|
| CRNNv1 (baseline) | 98.4 %   | 0.01555 |
| CRNNv2            | 95.4 %   | 0.04595    |


## Запуск
```bash
poetry install
poetry shell
python main.py
```

## Вывод модели
```bash
(ocr-emnist-py3.9)  ~/PycharmProjects/ocr-emnist/ [main] python main.py 
Generated Captcha: <PIL.Image.Image image mode=L size=140x28 at 0x160103550> with text 44501
Recognized Text: 44501
(ocr-emnist-py3.9)  ~/PycharmProjects/ocr-emnist/ [main] python main.py
Generated Captcha: <PIL.Image.Image image mode=L size=140x28 at 0x1624FE550> with text 41646
Recognized Text: 41646
(ocr-emnist-py3.9)  ~/PycharmProjects/ocr-emnist/ [main] python main.py
Generated Captcha: <PIL.Image.Image image mode=L size=140x28 at 0x15FB97580> with text 84119
Recognized Text: 86119

```


## Contributors
1. Шакиров Ренат
2. Набатчиков Илья
3. Могилевский Саша
