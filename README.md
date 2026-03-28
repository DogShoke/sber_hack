# Guardian of Truth Baseline

Рабочий baseline pipeline для хакатона:

`CSV -> features -> LogisticRegression -> PR-AUC -> single-example inference`

На текущем этапе реализована только базовая версия без hidden states, layer disagreement и sentence-level aggregation.

## Что делает baseline

1. Загружает `prompt`, `model_answer`, `is_hallucination` из CSV.
2. Делает один forward `ai-sage/GigaChat3-10B-A1.8B-bf16` на паре `prompt + model_answer`.
3. Извлекает базовые признаки:
   - длина ответа в символах
   - длина ответа в токенах
   - `mean logprob`
   - `min logprob`
   - `std logprob`
   - `mean entropy`
   - `max entropy`
4. Обучает `LogisticRegression`.
5. Считает `PR-AUC`, строит precision-recall curve и сохраняет артефакты.

## Структура проекта

```text
docs/
  guardian_of_truth_hackathon_task.md
notebooks/
  baseline.ipynb
data/
  knowledge_bench_public.csv
src/
  config.py
  data_utils.py
  evaluate.py
  features.py
  model_infer.py
  model_train.py
  utils.py
outputs/
  models/
  features/
  reports/
requirements.txt
README.md
```

## Organizer baseline.ipynb

Notebook организаторов делает более сложный baseline, чем текущая версия в `src/`:

- uncertainty features по токенам ответа
- internal scalar features из hidden states по нескольким слоям
- probe vector из последнего выбранного слоя
- `PCA + StandardScaler + LogisticRegression`

Текущая кодовая версия в `src/` намеренно упрощена до первого рабочего этапа.

## Установка

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Обучение

```bash
python -m src.model_train \
  --data-path data/knowledge_bench_public.csv \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --classifier lightgbm
```

## Быстрый цикл экспериментов

Если не хочется каждый раз заново прогонять `GigaChat`, можно разделить пайплайн на 2 этапа:

```bash
python -m src.extract_features \
  --data-path data/knowledge_bench_public.csv \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16
```

После этого можно быстро обучать разные табличные модели на уже сохраненных признаках:

```bash
python -m src.train_tabular --classifier lightgbm
python -m src.train_tabular --classifier logreg
```

Артефакты сохраняются в:

- `outputs/models/logreg_baseline.pkl`
- `outputs/features/baseline_features.npz`
- `outputs/reports/train_metrics.json`
- `outputs/reports/precision_recall_curve.csv`
- `outputs/reports/validation_predictions.csv`

## Инференс для одного примера

```bash
python -m src.model_infer \
  --model-bundle-path outputs/models/logreg_baseline.pkl \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --prompt "Кто открыл пенициллин?" \
  --model-answer "Пенициллин открыл Александр Флеминг."
```

## Отдельная оценка предсказаний

Если есть CSV с колонками `y_true`, `y_score` и опционально `inference_time_ms`:

```bash
python -m src.evaluate \
  --predictions-path outputs/reports/validation_predictions.csv
```

## Следующий этап

После фиксации этого baseline логично добавлять:

1. sentence-level features
2. hidden state features
3. layer disagreement features
