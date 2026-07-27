# House Prices - Advanced Regression Techniques

Проект решает задачу Kaggle **House Prices - Advanced Regression Techniques**.

Цель - предсказать цену продажи дома на основе табличных данных о характеристиках недвижимости.

## Описание задачи

Это задача регрессии.

Целевая переменная:

```text
SalePrice
```

Основная метрика соревнования — ошибка на логарифме цены:

```text
RMSE на log1p(SalePrice)
```

Поэтому модель обучается не на обычной цене, а на:

```python
np.log1p(SalePrice)
```

После предсказания результат переводится обратно:

```python
np.expm1(prediction)
```

## Структура проекта

```text
HousePrices/
├── configs/
│   └── config.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── notebooks/
│   └── eda.ipynb
├── results/
│   ├── all_model_results.csv
│   ├── final_model_metrics.csv
│   └── final_lasso_catboost_ensemble.joblib
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── submission.py
├── submissions/
│   └── final_lasso_catboost_ensemble_submission.csv
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Что было сделано

В проекте выполнены основные этапы ML-пайплайна:

- загрузка данных;
- первичный анализ данных;
- анализ пропущенных значений;
- анализ целевой переменной `SalePrice`;
- логарифмирование целевой переменной;
- поиск и удаление выбросов;
- feature engineering;
- обучение baseline-моделей;
- обучение линейных моделей;
- обучение boosting-моделей;
- обучение нейросетевой модели;
- построение ensemble-модели;
- создание финального Kaggle submission;
- перенос логики из notebook в модульный Python-код.

## Feature Engineering

Были добавлены дополнительные признаки:

```text
TotalSF
TotalBathrooms
TotalPorchSF
HouseAge
RemodAge
GarageAge
IsRemodeled
HasGarage
HasBasement
HasFireplace
HasPool
HasPorch
QualitySF
```

Также были удалены два выброса:

```text
Id = 524
Id = 1299
```

Эти объекты имели очень большую жилую площадь, высокое качество дома, но аномально низкую цену.

## Модели

В проекте были протестированы следующие модели:

- DummyRegressor;
- Ridge;
- Lasso;
- ElasticNet;
- RandomForestRegressor;
- GradientBoostingRegressor;
- LightGBM;
- XGBoost;
- CatBoost;
- MLPRegressor;
- Lasso + CatBoost Ensemble.

## Финальная модель

Финальная модель:

```text
Lasso + CatBoost VotingRegressor
```

Веса ансамбля:

```text
Lasso:    0.6
CatBoost: 0.4
```

Модель выбрана, потому что она показала лучший результат на cross-validation среди протестированных вариантов.

## Результаты

Локальный результат на cross-validation:

```text
CV RMSE log mean: 0.109384
CV RMSE log std:  0.006365
```

Kaggle Public Score:

```text
0.12861
```

## Как запустить проект

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить полный pipeline:

```bash
python main.py
```

## Что делает `main.py`

Файл `main.py` выполняет полный цикл:

1. загружает данные;
2. удаляет выбросы;
3. создаёт новые признаки;
4. разделяет признаки на числовые и категориальные;
5. создаёт preprocessing pipeline;
6. создаёт финальную ensemble-модель;
7. считает cross-validation score;
8. обучает модель на всех train-данных;
9. сохраняет метрики;
10. сохраняет обученную модель;
11. создаёт Kaggle submission;
12. проверяет корректность submission-файла.

## Выходные файлы

После запуска `main.py` создаются файлы:

```text
results/final_model_metrics.csv
results/final_lasso_catboost_ensemble.joblib
submissions/final_lasso_catboost_ensemble_submission.csv
```

## Финальный submission

Файл для загрузки на Kaggle:

```text
submissions/final_lasso_catboost_ensemble_submission.csv
```

## Используемые технологии

```text
Python
pandas
numpy
scikit-learn
CatBoost
LightGBM
XGBoost
matplotlib
seaborn
joblib
```

## Статус проекта

```text
EDA                  ✅
Feature Engineering  ✅
Modeling             ✅
Ensemble             ✅
DNN experiment       ✅
Kaggle submission    ✅
Modular src code     ✅
One-command pipeline ✅
```