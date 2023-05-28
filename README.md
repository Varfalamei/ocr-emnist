# OCR system for recognition handwritten characters

Домашнее задание к [задаче](https://github.com/aitalents/computer-vision-technology/tree/main/Topic%204.%20OCR)
и сделано оно по [этому](https://github.com/Alek-dr/OCR-Example/tree/master) репозиторию


## Пункты

- [x] архитектура реализована точно как описано в статье, а не просто похожа
- [x] выбрана и обоснована метрика качества
- [ ] модель обучена и запускается в демо режиме - т.е. есть скрипт, который загружает обученную модель, на лету генерируется капча, модель её распознает
- [x] есть понимание работы CTC loss
- [ ] ответы на вопросы
- [ ] студент сдал и защитил работу в кратчайший срок, на ближайшей официальной практике после лекции

## Метрика качества

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
gunicorn main:app -c gunicorn.config.py
```


## Contributors
1. Шакиров Ренат
2. Набатчиков Илья
3. Могилевский Саша
