# Assignment 3 — запуск и сдача

Задача вне основного плана, 2026-10-08. Реализация остаётся в существующем
репозитории и ветке `feature/assignment-3`. Новых runtime-зависимостей нет.
Исходный код моделей, обучение и старый preprocessing не изменены.

## Структура

| Файл | Назначение |
|---|---|
| [validation.py](../../src/eye_tracking_analysis/validation.py) | Строгий контракт CSV и временных меток |
| [analysis.py](../../src/eye_tracking_analysis/analysis.py) | Config/result, предобработка, события и признаки |
| [io.py](../../src/eye_tracking_analysis/io.py) | Загрузка, SHA256, CSV/JSON и защита старых runs |
| [visualization.py](../../src/eye_tracking_analysis/visualization.py) | Headless PNG без графического устройства |
| [cli.py](../../src/eye_tracking_analysis/cli.py) | CLI, явные единицы и параметры |
| [Пример](../../examples/eye_tracking_analysis_demo.py) | Детерминированная синтетическая геометрия |
| [Тесты](../../tests/test_eye_tracking_analysis.py) | Заранее известные результаты и граничные случаи |
| [CI](../../.github/workflows/ci.yml) | Ruff, core pytest, два примера, сборка; optional deep |
| [Английский отчёт](ASSIGNMENT_3_REPORT_EN.md) | Версионируемый редактируемый текст отчёта |
| [Issues](ISSUES.md) | Подготовленные тексты задач; не опубликованы |

## Установка и демонстрация

Из корня репозитория, Python 3.10+. Для новой среды:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e '.[dev]'
.venv/Scripts/python.exe examples/eye_tracking_analysis_demo.py
.venv/Scripts/python.exe -m src.eye_tracking_analysis.cli --help
```

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python examples/eye_tracking_analysis_demo.py
.venv/bin/python -m src.eye_tracking_analysis.cli --help
```

Пример создаёт новый UUID-каталог. Для анализа своего CSV:

```powershell
.venv/Scripts/python.exe -m src.eye_tracking_analysis.cli path/to/gaze.csv --output-dir outputs/new-analysis --coordinate-unit degrees --timestamp-unit ms --velocity-threshold 30
```

В Linux заменить исполняемый файл на `.venv/bin/python`. После установки
также доступен entry point `eye-tracking-analysis`. CSV требует явных IDs
участника, сессии, испытания и `gaze_x/gaze_y`; timestamp и blink опциональны.
`statement_id` сохраняется как дополнительное поле; каждое высказывание должно
иметь отдельный `trial_id`. Данные не сортируются для сокрытия ошибок времени.
Сброс времени допустим только в другой группе participant/session/trial.
Неизвестный класс не заполняется: модуль вообще не использует labels.

## Методы и ограничения

Пропущенной считается пара gaze, если отсутствует хотя бы одна координата.
Исходная пропущенность, число интерполяций и оставшиеся usable samples отражены
отдельно. Интерполяция по фактическому времени заполняет только целый внутренний
разрыв между двумя наблюдаемыми точками, если полный span не превышает заданный
лимит; края и длинные разрывы остаются пропусками. Blink-отсчёты не заполняются.
По умолчанию интерполяция и smoothing выключены.

Smoothing — среднее отсчётов в trailing-интервале `(t - window, t]`, заданном
в миллисекундах. Число точек зависит от реальных timestamps/частоты; это не
равномерная ресэмплинг-сетка и не частотно селективный физиологический фильтр.
На оставшихся пропусках и временных разрывах фильтр сбрасывается.
Нормализация при `screen_size` делит x/y на заранее известные width/height;
выход — normalized units. Pixel-to-degree conversion отсутствует: нужны геометрия
экрана и дистанция наблюдения. Данные не используются для обучения scaling.

Скорость интервала — `hypot(dx, dy) / dt`; low velocity (`<= threshold`)
образует оценку fixation, high velocity — оценку saccade. Длительность события —
разница крайних timestamps; амплитуда — расстояние между концами; mean velocity —
длина пути / длительность, peak — максимум интервалов. Короткие low-velocity
серии исключаются по `min_fixation_s`. Dispersion —
`sqrt(var(x, ddof=0) + var(y, ddof=0))` по usable обработанным точкам.
Все результаты относятся к испытанию, а не к независимому участнику.

Blink rate доступен только с полными наблюдениями blink 0/1 и временем:
`60 * число наблюдаемых 0->1 / сумму dt без больших acquisition gaps`.
Начало blink в первой строке/после gap неизвестно и не считается onset.
Отсутствие gaze не интерпретируется как blink. Без времени временные признаки
пустые; даже известная частота не подставляется вместо отсутствующих timestamps.
Пустой CSV отклоняется; полностью отсутствующий gaze даёт unavailable dispersion.

Модуль offline: интерполяция использует будущую точку. Он не доказывает
причинность/инвариантность пакетам, распознавание truth/lie или работу устройства.
Пороги инженерные; методы требуют валидации на разрешённой event-разметке.

## Выбор технологий и альтернативы

| Технология | Использование и причина | Альтернативы и причина выбора |
|---|---|---|
| Python | Уже используется; единый код анализа, тестов и ML | R удобен для статистики; MATLAB имеет signal toolbox, C++ — контроль latency. Перенос сейчас увеличивает сопровождение |
| NumPy | Разности, нормы, дисперсия, masks | Python lists медленнее для массивов; SciPy полезен для сложных фильтров, здесь не требуется |
| pandas | CSV, типы, группы, таблицы результатов | csv stdlib требует ручных групп; Polars полезен для крупных таблиц, миграция не нужна для малого модуля |
| Matplotlib | Воспроизводимый PNG через Agg | Plotly полезен для интерактивности; seaborn для статистических графиков. Здесь нужен автономный статический экспорт |
| pytest / pytest-cov | Границы, oracle-геометрия, интеграция, XML coverage | unittest встроен; hypothesis полезен для property tests. Сохраняется существующий pytest |
| Ruff | Ошибки импортов/имён и заданные правила стиля | Flake8/Pylint допустимы, добавление дублирующего инструмента не нужно |
| Git | Локальная история, ветка, review diff | SVN централизован; Mercurial распределён. Git уже настроен в проекте |
| GitHub / Actions | Существующий origin, Issues/PR и YAML CI | GitLab CI/Jenkins потребуют другую инфраструктуру. GitHub — платформа, Git — VCS |

scikit-learn, PyYAML, openpyxl остаются существующими зависимостями общего
проекта; PyTorch/XGBoost — optional. Новый модуль не вызывает классификаторы.
Скрипт создания отчёта использует bundled `python-docx`; это средство авторства,
не новая зависимость запуска анализа.

## Проверки

Свежая полная проверка вне песочницы: **97 passed in 56.89s**, в том числе
35 новых тестов; Ruff — `All checks passed!`, `pip check` —
`No broken requirements found`. Два optional deep-теста агентов и два legacy
deep-теста включены; это маленькие регрессионные runs, не тяжёлый эксперимент.
Итоговые записи проверок и ограничения сборки/документа — в
[WORK_STATUS](../WORK_STATUS.md). Локальная проверка YAML не подтверждает remote CI.

Для новой проверки выбирать новый basetemp, предварительно создав родителя:

```powershell
New-Item -ItemType Directory -Path outputs/assignment-3-check-MY-UNIQUE-ID
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp outputs/assignment-3-check-MY-UNIQUE-ID/pytest-temp
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m pip check
```

В Linux доступны обычные pytest temp directories. На этой Windows-машине
песочница снова дала WinError 5; повтор вне песочницы завершился успешно.
Старые временные каталоги не удалялись.

## Отчёт

Финальные документы: `outputs/assignment-3-report-20261008-03/Assignment_3_Report_EN.docx`
и такой же basename `.pdf`. Английский отчёт — 10 страниц, Times New Roman 12 pt,
интервал 1,5, основной текст по ширине, 13 проверенных источников.
PDF получен native экспортом отдельного скрытого Microsoft Word, затем
отрендерен Poppler; все страницы визуально проверены. LibreOffice отсутствует,
поэтому штатный render_docx.py не смог выполнить конвертацию.

Редактируемый исходник и средства сборки хранятся рядом в `docs/assignment_3/`.
Создание DOCX требует `python-docx` в среде авторства; в этом задании использован
bundled Python Codex, без изменения проектной `.venv`. Экспортный PowerShell
скрипт требует установленный Microsoft Word и открывает только указанный отчёт
для чтения. Команды из корня (выбрать новые выходные файлы):

```powershell
# python здесь — Python среды авторства с python-docx
python docs/assignment_3/build_report.py --output outputs/new-report/Assignment_3_Report_EN.docx
./docs/assignment_3/export_report_pdf.ps1 -DocumentPath outputs/new-report/Assignment_3_Report_EN.docx -PdfPath outputs/new-report/Assignment_3_Report_EN.pdf
```

В Linux можно конвертировать созданный DOCX имеющимся LibreOffice; этот путь
в текущей Windows-среде не проверен. Pending URLs оставлены явно: они не
выдаются за подтверждённые ссылки на опубликованный revision/remote run.

## Публикация и дальнейшие шаги

MIT уже существует; её условия не переоформлялись. Внешние данные сохраняют
свои условия доступа. Raw/processed, outputs, среды и секреты исключены из Git.
Пользовательская правка WORK_STATUS сохраняется вне автоматических коммитов.

После отдельного разрешения: push ветки, PR в main, публикация подготовленных
Issues, проверка реального GitHub run и ссылка на него в отчёте. Release требует
отдельного намерения и `ENABLE_RELEASE=true`; для сдачи release не нужен.
Pending revision/run URLs в отчёте заполнить только подтверждёнными ссылками.

Для диссертации: строгие labels в старом обучении, подтверждённые truth/lie-данные
и события, причинный pipeline и chunk-invariance, групповые сравнения baseline/NN,
измерения накопления/вычисления, затем физический источник. Этот assignment не
закрывает соответствующие пункты основного плана.
