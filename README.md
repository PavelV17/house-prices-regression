# House Prices - Advanced Regression Techniques

Проект решает Kaggle-задачу **House Prices - Advanced Regression Techniques**: предсказать цену продажи дома по табличным признакам недвижимости.

Основной production-style pipeline находится в `main.py` и модуле `src/`. Ноутбук `notebooks/eda.ipynb` используется для EDA, визуализаций, проверки гипотез и запуска экспериментов из одного места.

---

## 1. Тип задачи и метрика

Это задача **регрессии**.

Целевая переменная:

```text
SalePrice
```

Основная Kaggle-метрика соревнования:

```text
RMSE на log1p(SalePrice)
```

Поэтому модель обучается на логарифме цены:

```python
np.log1p(SalePrice)
```

После предсказания результат переводится обратно в деньги:

```python
np.expm1(prediction_log)
```

Дополнительно считаются бизнес-метрики:

| Metric | Meaning |
|---|---|
| `MAE price` | средняя абсолютная ошибка в деньгах |
| `RMSE price` | ошибка в деньгах, где большие ошибки штрафуются сильнее |
| `MAPE` | средняя процентная ошибка |
| `WAPE` | суммарная абсолютная ошибка / сумма реальных цен |

---

## 2. Validation, CV и test

В проекте есть три разных уровня оценки.

| Level | Для чего используется | Что означает |
|---|---|---|
| Cross-validation / OOF | выбор модели и гиперпараметров | внутренняя оценка на `train.csv` |
| Kaggle Public Score | внешняя проверка на скрытых ответах Kaggle | проверка на test set соревнования |
| PyTorch validation split | отдельный DL-эксперимент | не сравнивается напрямую с CV-таблицей |

Важно: **CV-результат не называется финальным unbiased test quality**. Он используется для model selection. Внешняя проверка качества — это Kaggle Public Score.

---

## 3. Как получить данные

Данные не хранятся в репозитории. Их нужно скачать с Kaggle:

```text
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
```

После скачивания положите файлы в папку `data/`:

```text
HousePrices/
└── data/
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    └── data_description.txt
```

Также можно скачать через Kaggle CLI:

```bash
mkdir data
kaggle competitions download -c house-prices-advanced-regression-techniques -p data --unzip
```

---

## 4. Установка

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Быстрый запуск

Основной pipeline:

```bash
python main.py --config configs/config.yaml --experiment fixed_features
```

Stacking ensemble:

```bash
python main.py --config configs/stacking.yaml --experiment stacking
```

PyTorch MLP baseline:

```bash
python run_torch_mlp.py --config configs/config.yaml
```

Optuna tuning:

```bash
python run_optuna.py --config configs/config.yaml --trials 100 --experiment optuna_100
```

Запуск лучшего сохранённого Optuna-конфига:

```bash
python main.py --config results/optuna_best_config.yaml --experiment optuna_100_best
```

---

## 6. Что делает `main.py`

`main.py` выполняет полный цикл:

1. загружает `train.csv`, `test.csv`, `sample_submission.csv`;
2. применяет raw data fixes;
3. удаляет заранее найденные выбросы только из train;
4. создаёт новые признаки;
5. делит признаки на числовые и категориальные;
6. строит preprocessing pipeline;
7. создаёт модель;
8. считает cross-validation и OOF-метрики;
9. обучает финальную модель на всех очищенных train-данных;
10. сохраняет модель и метрики;
11. создаёт Kaggle submission;
12. записывает запуск в `results/experiments.csv`.

---

## 7. Структура проекта

```text
HousePrices/
├── configs/
│   ├── config.py
│   ├── config.yaml
│   └── stacking.yaml
├── data/                         # не хранится в git
├── notebooks/
│   └── eda.ipynb
├── results/
│   ├── all_model_results.csv
│   ├── experiments.csv
│   ├── final_model_metrics.csv
│   ├── optuna_best_config.yaml
│   ├── optuna_best_trial.csv
│   └── torch_mlp_metrics.csv
├── src/
│   ├── data.py
│   ├── features.py
│   ├── inference.py
│   ├── models.py
│   ├── submission.py
│   ├── torch_mlp.py
│   └── train.py
├── submissions/
│   └── final_lasso_catboost_ensemble_submission.csv
├── main.py
├── run_torch_mlp.py
├── run_optuna.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 8. Что исправлено после ревью

### 8.1. `MSSubClass` теперь категориальный признак

`MSSubClass` выглядит как число, но по смыслу это код типа дома. Поэтому он не обрабатывается как непрерывный числовой признак.

Настройка в `configs/config.yaml`:

```yaml
preprocessing:
  categorical_numeric_features: [MSSubClass]
```

В коде это применяется через `src/features.py -> apply_raw_data_fixes()`.

### 8.2. Обработаны невозможные значения `GarageYrBlt`

`GarageYrBlt` — год постройки гаража. Значения вроде `2207` невозможны. Также гараж не должен быть построен позже года продажи дома.

Настройка:

```yaml
preprocessing:
  invalid_numeric_values:
    GarageYrBlt:
      min: 1800
      max_relative_to_column: YrSold
      replacement: null
```

Логика:

```text
GarageYrBlt < 1800   -> NaN
GarageYrBlt > YrSold -> NaN
```

После этого пропуски обрабатываются imputer-ом внутри sklearn pipeline.

### 8.3. Validation без лишнего двойного обучения

Раньше оценка могла запускать `cross_val_score` и `cross_val_predict` отдельно. Это приводило к двойному числу fit-ов для одной и той же CV-схемы.

Теперь `src/models.py -> evaluate_model()` использует один явный CV-loop:

```text
fit on fold train
predict fold validation
save fold RMSE
save OOF predictions
```

Так fold scores и OOF-метрики считаются на одной и той же разметке фолдов, а оценка работает быстрее.

### 8.4. Добавлен stacking ensemble

Stacking обучает meta-model поверх OOF-предсказаний Lasso и CatBoost.

Запуск:

```bash
python main.py --config configs/stacking.yaml --experiment stacking
```

### 8.5. Добавлен Optuna tuning

Optuna подбирает параметры `Lasso`, `CatBoost` и веса ансамбля. Search space сфокусирован вокруг уже сильной baseline-модели, чтобы не тратить время на явно слабые области.

Подбираются:

```text
lasso_alpha
catboost_iterations
catboost_learning_rate
catboost_depth
catboost_l2_leaf_reg
ensemble_lasso_weight
```

Также добавлен фиксированный `TPESampler(seed=random_state)`, чтобы подбор был более воспроизводимым.

---

## 9. Feature Engineering

Добавленные признаки:

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

Удаляются два заранее найденных train-выброса:

```text
Id = 524
Id = 1299
```

Причина: эти объекты имеют очень большую площадь и высокое качество дома, но аномально низкую цену.

---

## 10. Model selection results

Эта таблица показывает результаты **cross-validation**. Она используется для выбора модели и гиперпараметров.

| Model | CV RMSE log mean | CV RMSE log std | Notes |
|---|---:|---:|---|
| Optuna-tuned Lasso + CatBoost | 0.108579 | 0.007057 | лучший локальный CV candidate |
| Stacking Lasso + CatBoost | 0.108672 | 0.006695 | лучший до Optuna |
| Lasso + CatBoost FE + data fixes | 0.108700 | 0.006542 | после исправлений ревью |
| Lasso + CatBoost FE baseline | 0.109384 | 0.006365 | до исправлений ревью |
| Mixed Ensemble FE | 0.109484 | 0.006188 | ensemble |
| Lasso FE | 0.112641 | 0.006404 | feature engineering |
| CatBoost FE | 0.113077 | 0.007296 | boosting |
| XGBoost FE | 0.115225 | 0.005817 | boosting |
| LightGBM FE | 0.123803 | 0.007862 | boosting |
| RandomForest FE | 0.133820 | 0.008544 | tree baseline |
| DummyRegressor | 0.399285 | 0.016812 | dummy baseline |

---

## 11. Текущий лучший локальный candidate

Текущий лучший локальный candidate после Optuna:

```text
Optuna-tuned Lasso + CatBoost weighted ensemble
```

Параметры сохранены в:

```text
results/optuna_best_config.yaml
```

Метрики после запуска сохранённого Optuna-конфига:

| Metric | Value |
|---|---:|
| CV RMSE log mean | 0.108579 |
| CV RMSE log std | 0.007057 |
| OOF RMSE log | 0.108812 |
| OOF MAE price | 12912.27 |
| OOF RMSE price | 19966.24 |
| OOF MAPE | 7.57% |
| OOF WAPE | 7.14% |

Параметры модели:

| Parameter | Value |
|---|---:|
| Lasso alpha | 0.000565 |
| CatBoost iterations | 500 |
| CatBoost learning rate | 0.049955 |
| CatBoost depth | 5 |
| CatBoost L2 leaf reg | 2.848276 |
| Lasso weight | 0.567576 |
| CatBoost weight | 0.432424 |

---

## 12. External evaluation: Kaggle Public Score

Последний подтверждённый Kaggle Public Score:

| Submission | Kaggle Public Score |
|---|---:|
| Lasso + CatBoost / stacking candidate after review fixes | 0.12835 |

После Optuna создан новый `submission.csv`, но его внешний Kaggle score нужно проверить отдельной отправкой на Kaggle.

---

## 13. PyTorch MLP baseline

В проект добавлен отдельный PyTorch baseline:

```text
src/torch_mlp.py
```

Он использует:

```text
TensorDataset / DataLoader
MLPRegressorNet
MSELoss
AdamW
early stopping
train / validation loop
```

Запуск:

```bash
python run_torch_mlp.py --config configs/config.yaml
```

Последний PyTorch MLP эксперимент на validation split:

| Metric | Value |
|---|---:|
| Validation RMSE log | 0.146650 |
| Validation MAE price | 19157.68 |
| Validation RMSE price | 25747.76 |
| Validation MAPE | 11.17% |
| Validation WAPE | 10.56% |
| Epochs ran | 201 / 250 |

Это не cross-validation результат, а отдельный validation split. Поэтому PyTorch MLP не смешивается напрямую с CV-таблицей классических моделей.

---

## 14. Inference-only запуск

После обучения можно не переобучать модель, а только создать submission из сохранённой модели:

```bash
python main.py --inference-only
```

Этот режим требует, чтобы уже существовал файл:

```text
results/final_lasso_catboost_ensemble.joblib
```

---

## 15. Выходные файлы

После `python main.py` создаются:

```text
results/final_model_metrics.csv
results/final_lasso_catboost_ensemble.joblib
results/experiments.csv
submissions/final_lasso_catboost_ensemble_submission.csv
```

После `python run_torch_mlp.py`:

```text
results/torch_mlp_metrics.csv
```

После `python run_optuna.py`:

```text
results/optuna_trials.csv
results/optuna_best_config.yaml
```

Файл для загрузки на Kaggle:

```text
submissions/final_lasso_catboost_ensemble_submission.csv
```

---

## 16. Статус проекта

```text
EDA                                      ✅
Feature Engineering                      ✅
Modeling                                 ✅
Weighted Ensemble                        ✅
Stacking Ensemble                        ✅
PyTorch MLP baseline                     ✅
MSSubClass categorical fix               ✅
GarageYrBlt invalid value fix            ✅
Optuna tuning                            ✅
Kaggle submission                        ✅
Modular src code                         ✅
One-command pipeline                     ✅
Pinned requirements                      ✅
YAML experiment config                   ✅
Experiment tracking CSV                  ✅
Business metrics MAPE/WAPE               ✅
```
