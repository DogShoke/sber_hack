# Guardian of Truth

Детектор галлюцинаций для хакатона, построенный по схеме:

`prompt + model_answer -> один forward GigaChat -> rich internal features -> tabular detector`

Основная идея проекта: выжимать максимум из внутренних сигналов `ai-sage/GigaChat3-10B-A1.8B-bf16`, а сверху держать легкий классификатор и его ансамбли. Это соответствует ограничению задания: анализ hidden states и активаций автора галлюцинаций допустим, а отдельные внешние runtime-модели нежелательны.

## Текущее состояние

- Рабочий rich-internal extractor в [`C:\Hack\src\features.py`](C:\Hack\src\features.py)
- Поддержка `4-bit` / `8-bit` загрузки и offload для Colab и ограниченных GPU
- Обучение `LightGBM` и `LogisticRegression` на cached features
- Ансамбль и stacking поверх уже извлеченных признаков
- Merge внешних feature dumps для offline-обучения
- Docker-упаковка для воспроизводимого запуска

Лучший устойчивый preview-only результат на текущем этапе:

- `300/300 preview split`
- `LightGBM + LogReg`, веса `0.5 / 0.5`
- `PR-AUC = 0.756018`

`TruthfulQA` уже проверялся как внешний датасет: одиночный `LightGBM` слегка улучшался, но финальный ансамбль лучше не стал. Значит внешний train set нужно подбирать ближе к distribution целевой задачи.

## Ветки

- `main`
  Текущая стабильная ветка. Здесь должен лежать рабочий пайплайн, который можно запускать локально, в Colab и через Docker.
- `codex/colab-4bit`
  Историческая ветка, в которой отлаживалась загрузка модели в `4-bit` для бесплатного Colab T4.
- `codex/rich-internal-detector`
  Основная исследовательская ветка с richer internal features, ансамблями, tuning, stacking и merge внешних feature dumps. На момент обновления README именно она является источником текущего лучшего пайплайна и должна быть слита в `main`.

## Формат данных

Все входные CSV приводятся к трем колонкам:

- `prompt`
- `model_answer`
- `is_hallucination`

Поддерживаемые значения метки:

- `0` / `1`
- `false` / `true`

## Структура проекта

```text
data/
  knowledge_bench_public.csv
docs/
  guardian_of_truth_hackathon_task.md
notebooks/
  baseline.ipynb
outputs/
  features/
  models/
  reports/
src/
  config.py
  data_utils.py
  ensemble_tabular.py
  evaluate.py
  extract_features.py
  features.py
  merge_feature_dumps.py
  model_infer.py
  model_train.py
  stack_tabular.py
  train_tabular.py
  tune_tabular.py
  utils.py
.dockerignore
.gitignore
Dockerfile
README.md
requirements.txt
```

## Основные сценарии

### 1. Полный train за один запуск

Подходит для небольших smoke tests и базовых прогонов.

```bash
python -m src.model_train \
  --data-path data/knowledge_bench_public.csv \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --classifier lightgbm
```

Полезные флаги для Colab и ограниченных GPU:

```bash
--load-in-4bit
--offload-folder offload
```

### 2. Рекомендуемый режим: extract once, train many

Сначала один раз извлекаем признаки:

```bash
python -m src.extract_features \
  --data-path data/knowledge_bench_public.csv \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --feature-dump-path outputs/features/baseline_features.npz
```

Для Colab:

```bash
python -m src.extract_features \
  --data-path data/knowledge_bench_public.csv \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --feature-dump-path outputs/features/baseline_features.npz \
  --load-in-4bit \
  --offload-folder /content/offload
```

Потом быстро обучаем табличные модели на готовых признаках:

```bash
python -m src.train_tabular \
  --classifier lightgbm \
  --feature-dump-path outputs/features/baseline_features.npz \
  --output-model-path outputs/models/lightgbm.pkl \
  --report-path outputs/reports/lightgbm_metrics.json \
  --curve-path outputs/reports/lightgbm_curve.csv

python -m src.train_tabular \
  --classifier logreg \
  --feature-dump-path outputs/features/baseline_features.npz \
  --output-model-path outputs/models/logreg.pkl \
  --report-path outputs/reports/logreg_metrics.json \
  --curve-path outputs/reports/logreg_curve.csv
```

### 3. Ансамбль табличных моделей

```bash
python -m src.ensemble_tabular \
  --feature-dump-path outputs/features/baseline_features.npz \
  --bundle-paths outputs/models/lightgbm.pkl outputs/models/logreg.pkl \
  --weights 0.5 0.5 \
  --report-path outputs/reports/ensemble_metrics.json \
  --curve-path outputs/reports/ensemble_curve.csv
```

### 4. Hyperparameter tuning

Практический вариант: random search.

```bash
python -m src.tune_tabular \
  --classifier lightgbm \
  --search-type random \
  --n-iter 32 \
  --feature-dump-path outputs/features/baseline_features.npz \
  --output-model-path outputs/models/lightgbm_tuned.pkl \
  --report-path outputs/reports/lightgbm_tuned_metrics.json \
  --curve-path outputs/reports/lightgbm_tuned_curve.csv
```

### 5. Stacking

```bash
python -m src.stack_tabular \
  --feature-dump-path outputs/features/baseline_features.npz \
  --report-path outputs/reports/stacked_metrics.json \
  --curve-path outputs/reports/stacked_curve.csv
```

### 6. Работа с внешними датасетами

Для внешнего датасета сначала извлекаем признаки в отдельный dump:

```bash
python -m src.extract_features \
  --data-path data/external_dataset.csv \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --feature-dump-path outputs/features/external_features.npz
```

Потом объединяем с preview dump:

```bash
python -m src.merge_feature_dumps \
  --input-paths outputs/features/external_features.npz outputs/features/baseline_features.npz \
  --output-path outputs/features/merged_features.npz \
  --validation-source last
```

И уже на merged dump обучаем модели и ансамбль.

## Инференс для одного примера

```bash
python -m src.model_infer \
  --model-bundle-path outputs/models/lightgbm.pkl \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --prompt "Кто открыл пенициллин?" \
  --model-answer "Пенициллин открыл Александр Флеминг."
```

Для Colab и слабых GPU:

```bash
python -m src.model_infer \
  --model-bundle-path outputs/models/lightgbm.pkl \
  --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
  --prompt "Кто открыл пенициллин?" \
  --model-answer "Пенициллин открыл Александр Флеминг." \
  --load-in-4bit \
  --offload-folder offload
```

## Артефакты

Основные артефакты по умолчанию:

- `outputs/features/*.npz`
- `outputs/models/*.pkl`
- `outputs/reports/*.json`
- `outputs/reports/*.csv`

## Docker

В проекте есть [`C:\Hack\Dockerfile`](C:\Hack\Dockerfile) и [`C:\Hack\.dockerignore`](C:\Hack\.dockerignore).

### Сборка образа

```bash
docker build -t guardian-of-truth .
```

### Интерактивный запуск

Если на хосте настроен NVIDIA Container Toolkit:

```bash
docker run --rm -it --gpus all -v ${PWD}:/app guardian-of-truth bash
```

### Пример extraction в контейнере

```bash
docker run --rm -it --gpus all -v ${PWD}:/app guardian-of-truth \
  python3 -m src.extract_features \
    --data-path data/knowledge_bench_public.csv \
    --model-name-or-path ai-sage/GigaChat3-10B-A1.8B-bf16 \
    --feature-dump-path outputs/features/baseline_features.npz \
    --load-in-4bit \
    --offload-folder /app/offload
```

### Пример обучения в контейнере

```bash
docker run --rm -it --gpus all -v ${PWD}:/app guardian-of-truth \
  python3 -m src.train_tabular \
    --classifier lightgbm \
    --feature-dump-path outputs/features/baseline_features.npz \
    --output-model-path outputs/models/lightgbm.pkl \
    --report-path outputs/reports/lightgbm_metrics.json \
    --curve-path outputs/reports/lightgbm_curve.csv
```

## Что дальше

Если нужно поднимать качество выше текущего `~0.75-0.76` на preview-only dev, самый перспективный следующий шаг:

- искать более близкие внешние hallucination datasets
- извлекать на них те же rich internal features
- дообучать offline на merged feature dumps

Сейчас наибольший риск не в недостатке табличных моделей, а в domain shift внешних данных.
