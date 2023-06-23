# ADA

## Описание
Пример оформления заданий по оптимизационным задачам машинного обучения
## Запуск

```bash
git clone https://github.com/DAAksenov/loc-extr-main.git
```
```bash
cd ./loc-extr-main/
streamlit run main.py
```
```
Требуется версия Python 3.9, protobuf==3.20 (даунгрейд после установки requirements)
```

```bash
docker build --tag ada .
docker run -d -p 8501:8501 ada
```
