# Контекст и карта проекта

Обновлено: 2026-10-08. Оперативное состояние — в [WORK_STATUS.md](WORK_STATUS.md).

## Цель

Пользователь разрабатывает магистерскую диссертацию и хочет реально рабочий инструмент. Приложенный обзор: «A Systematic Review of Neural Network Analysis of Real-Time Eye-Tracking Data for Detecting Deceptive Statements». Это тематическая основа, не техническое задание и не подтверждение официального названия кафедрой.

Целевой сценарий: запись/поток -> качество -> причинная обработка -> окно -> сохранённая нейросеть -> оценка/статус -> отчёт. Научная часть отдельно проверяет новых участников, задержки и устойчивость. Сейчас существует офлайн-прототип обучения. Основной предмет исследования — truth/lie (выбор пользователя 2026-09-30, D-005). CIT — отдельный target; его метки не объединяются с truth/lie. Источник truth/lie данных, основной протокол и единица решения ещё требуют определения.

## Карта

| Область | Файлы | Назначение |
|---|---|---|
| Данные | `src/data/schema.py`, `adapters.py`, `loaders.py`, `synthetic.py` | Схема, XLSX/CSV, synthetic |
| Обработка | `src/features/preprocessing.py` | Валидность, интерполяция, smoothing, normalization |
| Assignment 3 | `src/eye_tracking_analysis/`, `examples/eye_tracking_analysis_demo.py` | Строгий label-free CSV, participant/session/trial, offline event estimates, quality, график, защищённый экспорт |
| Признаки | `src/features/aggregated.py`, `windowing.py` | Trial aggregates и окна внутри trial |
| Модели | `src/models/baselines.py`, `sequence.py` | Классические модели, LSTM, TCN |
| Обучение | `src/training/pipelines.py`, `trainer.py`, `datasets.py` | Splits, training, checkpoints |
| Оценка | `src/evaluation/metrics.py`, `plots.py` | Метрики и графики |
| Инференс | `src/inference/streaming.py`, `src/agents/services.py` | Агрегатор готовых вероятностей; label-free offline inference сохранённых baseline/LSTM/TCN через ModelSpec |
| Агентный слой | `src/agents/`, `configs/multi_agent.yaml`, `examples/multi_agent_demo.py` | Шесть агентов, Orchestrator, состояние, quality/verification/report, CLI и журнал |
| Вход | `scripts/train.py`, `main.py` | Реальный CLI обучения; main — заглушка |
| Проверки | `tests/`, `pyproject.toml`, `.github/workflows/ci.yml` | pytest, Ruff, build/release |

Конфигурации: `configs/synthetic_baseline.yaml`, `cit_baseline.yaml`, `cit_lstm.yaml`.

## Среда и команды

Текущая машина: Windows PowerShell, `D:/DeceptionDetection`. В коде использовать переносимые пути. На 2026-09-30 в `.venv` проверены Python 3.12.4, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, pytest 9.1.1, Ruff 0.16.9. Это снимок, не lockfile и не подтверждение deep-зависимостей. Требования задаёт pyproject; сохранять заявленную поддержку Python 3.10+ и pandas 2/3, пока она явно не изменена.

Из корня проекта:

```powershell
# Установка только при необходимости:
.venv/Scripts/python.exe -m pip install -e '.[dev]'
# Deep требует отдельного extra:
.venv/Scripts/python.exe -m pip install -e '.[dev,deep]'

.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m pytest -m 'not deep'
.venv/Scripts/python.exe -m pytest -m deep
.venv/Scripts/python.exe -m pytest --cov=src --cov-report=term-missing
.venv/Scripts/python.exe scripts/train.py --config configs/synthetic_baseline.yaml
.venv/Scripts/python.exe examples/multi_agent_demo.py
.venv/Scripts/python.exe examples/multi_agent_demo.py --open
.venv/Scripts/python.exe -m src.agents.cli --help
.venv/Scripts/python.exe examples/eye_tracking_analysis_demo.py
.venv/Scripts/python.exe -m src.eye_tracking_analysis.cli --help
.venv/Scripts/python.exe -m build
```

Synthetic training CLI пишет в `outputs/synthetic_demo` и может перезаписать demo. Для исследования копировать config с уникальным output_dir. Агентный пример создаёт новый UUID-каталог; production agent CLI требует реальный файл и доверенный checkpoint/ModelSpec и не обучает модель. Физический live/replay engine ещё не реализован. Инструкция и ограничения: [MULTI_AGENT_ARCHITECTURE_RU.md](MULTI_AGENT_ARCHITECTURE_RU.md).

Системный Python ранее получил PermissionError в pytest temp/cache; повтор с новым basetemp в test-results тоже не помог. В случае повторения проверять права и окружение. На 2026-09-30 полный запуск через `.venv` вне песочницы с отключённым cacheprovider прошёл (22 passed); внутри песочницы PermissionError сохраняется. Для явного basetemp заранее создавать его родительский каталог и выбирать новый путь для каждого запуска. Можно отключить cacheprovider и выбрать новый доступный basetemp, но не выдавать обход за исправление алгоритма и не использовать каталог с ценными файлами.

## Подтверждённые ограничения

- Пользователь 2026-09-30 подтвердил отсутствие truth/lie-данных и собственного айтрекера. Bag-of-Lies — кандидат с доступом по соглашению, не полученный датасет; результаты поиска — в [DATASET_SEARCH_RU.md](DATASET_SEARCH_RU.md). Replay можно проверять без устройства, физический live остаётся непроверенным.

- Обнаружено 16 локальных CIT XLSX; число участников, labels, полнота и timestamp units не аудитированы.
- В старом обучающем пути missing labels могут стать 0; CIT может угадывать класс по filename. Новый агентный inference удаляет labels до обработки и не вызывает fallback; training пока не исправлен.
- Trials выделяются сменой stimulus name; нужна сверка с протоколом/событиями.
- Валидность и pupil могут преимущественно использовать левый глаз.
- `online_safe` делает причинным только smoothing; прочая обработка полной записи не гарантирует онлайн-готовность.
- Fixation count — эвристика малых перемещений, не проверенный event detector.
- Baseline validation split не используется для подбора; within-subject split может разделять overlapping windows.
- Deep checkpoint хранит только веса; законченного inference bundle нет. Агентный inference требует отдельный явный ModelSpec с архитектурой, каналами, preprocessing, units, target/classes и SHA256; совпадение старого preprocessing с обучением зависит от корректности декларации.
- Bag-of-Lies adapter — placeholder; replay/live engine и SDK отсутствуют. Есть автономный HTML-показ сохранённых synthetic запусков; рабочий интерфейс физического/live исследования не реализован.
- TCN обрезает результат после двух свёрток; причинность промежуточных выходов ещё нужно проверить.

## Документы

- [AGENTS.md](../AGENTS.md) — инструкции агентам.
- [WORK_STATUS.md](WORK_STATUS.md) — состояние и очередь.
- [DECISIONS.md](DECISIONS.md) — решения/предложения.
- [План диссертации](THESIS_IMPLEMENTATION_PLAN_RU.md) — этапы и приёмка.
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — известные ограничения.
- [Агентная архитектура](MULTI_AGENT_ARCHITECTURE_RU.md) — дополнительный слой вне плана, CLI и контракты.
- [Наглядный показ](DEMO_GUIDE_RU.md) — запуск одной командой, автономный HTML и сценарий объяснения.
- [Шаблон эксперимента](templates/EXPERIMENT_RECORD.md) — воспроизводимость.
- [Assignment 3](assignment_3/GUIDE_RU.md) — отдельное учебное поручение, новые offline методы и 10-страничный английский отчёт; основной научный план не закрыт.
- PROJECT_REPORT и ASSIGNMENT_RU — исторические материалы учебного задания; не полный план диссертации.

PDF вне репозитория: `C:/Users/LEGION/Downloads/Springer_Nature_LaTeX_Template__1_.pdf`; на другой машине его наличие не гарантировано. План содержит извлечённый предметный контекст. Только подтверждённое поведение описывать как реализованное.
