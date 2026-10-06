# Состояние и очередь

Обновлено: 2026-10-06. Обновлять после существенной задачи; не хранить стенограмму.

## Последняя задача

Подготовлен наглядный показ агентного слоя (2026-10-06): автономный HTML с
пошаговым просмотром сохранённых agent executions, probability outputs,
quality, признаками, журналом, распределением calls и скачиванием JSON report.
Добавлены `src/agents/presentation.py`, `presentation.html`,
`examples/show_demo.cmd`, `tests/test_agent_presentation.py` и [DEMO_GUIDE_RU.md](DEMO_GUIDE_RU.md).
Существующий пример получил `--present`/`--open`: обучает baseline один раз,
реально запускает нормальный сигнал, плохой synthetic сигнал и неверный checksum,
затем объединяет их artifacts. HTML не выполняет inference и не изображает live.
Runtime-зависимости и научные алгоритмы не изменены; UI-01/STREAM-01 не закрыты.

Готовый файл: `outputs/agent-presentation-20261006-01/presentation-v2.html`;
открытие default browser через Python webbrowser вернуло True. Новый показ:
`.venv/Scripts/python.exe examples/multi_agent_demo.py --open` или двойной клик
`examples/show_demo.cmd`. Для просмотра достаточно одного HTML, без Python/сервера.

Проверки текущей задачи:

- `.venv/Scripts/python.exe examples/multi_agent_demo.py --present --output-dir outputs/agent-presentation-20261006-01` — exit 0; workflow COMPLETED, два expected FAILED, artifacts сохранены.
- `.venv/Scripts/python.exe -m pytest tests/test_agent_presentation.py tests/test_agents.py -p no:cacheprovider --basetemp outputs/presentation-check-20261006-01/pytest-temp` вне песочницы — **38 passed in 14.58s**, exit 0. Проверены реальное выполнение трёх сценариев, safe embedding JSON, запрет перезаписи viewer и регрессии агентного слоя.
- `.venv/Scripts/python.exe -m ruff check .` — All checks passed, exit 0. Полный/deep suite для изменений viewer не повторялся; предыдущие 60 passed относятся к предыдущей реализации ниже.
- `node outputs/presentation-check-20261006-01/browser-check.cjs` вне песочницы — exit 0, Edge 154.0.4258.53: initial/step/finish, play/pause/reset, три сценария, фактический download JSON, zero JS exceptions, нет horizontal overflow при 390px. Screenshot desktop/mobile и browser-check.json в том же каталоге. Внутри песочницы headless Edge не запускался; во время browser QA найдена и исправлена ошибка обращения к progress span, затем проверка прошла.
- `.venv/Scripts/python.exe -m build --no-isolation --outdir outputs/presentation-check-20261006-01/dist` вне песочницы — exit 0, sdist/wheel с HTML template и launcher собраны. `git diff --check` — exit 0.
- `.venv/Scripts/python.exe outputs/presentation-check-20261006-01/verify_delivery.py` вне песочницы — exit 0: 6 Markdown-файлов/33 локальные ссылки, template соответствует исходнику внутри wheel, launcher присутствует в sdist.

Synthetic example демонстрирует Logistic Regression, не нейросеть/физический
айтрекер. Встроенные результаты и анимация не подтверждают научную accuracy
или causality. Перед показом достаточно подготовить HTML; для нового вычисления
нужна установленная среда. Следующий научный шаг остаётся SPEC-01/DATA-02.

## Предыдущая реализация multi-agent слоя

Выполнено отдельное поручение вне основного плана: multi-agent слой в `src/agents/`
с шестью агентами и Orchestrator, типизированными сообщениями/состоянием,
quality/verification/report, ограниченными retries/steps, кооперативным дедлайном,
журналом agent/tool calls и статистикой. CLI применяет сохранённые baseline/LSTM/TCN,
не обучает и не получает true labels. Источник — существующие CSV/CIT XLSX adapters;
выбирается одно явно ограниченное испытание. Добавлены `configs/multi_agent.yaml`,
`examples/multi_agent_demo.py`, `tests/test_agents.py`, `tests/test_agents_deep.py`,
console entry point. Совместимые параметры добавлены в adapters и aggregate extractor;
алгоритмы preprocessing/features/models не переписаны. README/KNOWN_ISSUES,
PROJECT_CONTEXT и DECISIONS дополнены. Архитектура, таблица ролей, diagram, команды
и ограничения: [MULTI_AGENT_ARCHITECTURE_RU.md](MULTI_AGENT_ARCHITECTURE_RU.md), D-007.

Проверки финального кода на 2026-10-06:

```powershell
New-Item -ItemType Directory -Path outputs/agents-check-20261006-04
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp outputs/agents-check-20261006-04/pytest-temp --junitxml outputs/agents-check-20261006-04/junit.xml
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m build --no-isolation --outdir outputs/agents-check-20261006-04/dist
```

Pytest вне песочницы: **60 passed in 43.86s**, exit 0 (22 прежних + 38 новых,
включая LSTM/TCN). Ruff в песочнице: All checks passed, exit 0. Build вне песочницы:
sdist и wheel собраны, exit 0. Typecheck в проекте не настроен, новый инструмент
не добавлялся. Перед build установлена недостающая `wheel==0.48.0` в `.venv`;
она уже объявлена в build-system, runtime-зависимости не изменены.

Дополнительно: `.venv/Scripts/python.exe outputs/agents-check-20261006-04/verify_delivery.py`
вне песочницы — exit 0; проверены 6 Markdown-файлов/28 локальных ссылок,
соответствие исходников agents/aggregator внутри wheel, console entry point,
source hash, split separation и реальные demo artifacts. Сводка:
`outputs/agents-check-20261006-04/delivery-check.json`. `git diff --check` — exit 0.

Документированный пример: `.venv/Scripts/python.exe examples/multi_agent_demo.py --output-dir outputs/agent-demo-20261006-01`
в песочнице — exit 0, COMPLETED, quality ACCEPTED, verification INCONCLUSIVE.
Сохранены JSON artifacts, model/source hashes, config, seed 42, split manifest
и запись по шаблону в `outputs/agent-demo-20261006-01/EXPERIMENT_RECORD.md`.
Проверены 14 вызовов: Orchestrator 1 (7.14%), FeatureExtractionAgent 3 (21.43%),
остальные пять агентов по 2 (14.29%); максимум ниже 40%. Это доли вызовов, не CPU.

Первый новый pytest внутри песочницы снова получил WinError 5 для собственного
basetemp; build также получил PermissionError temp. Старые временные каталоги
не удалялись; использованы новые отдельные output-пути и проверка вне песочницы.
Это ограничение среды, не исправленные права или дефект ML.

Ограничения: текущая обработка **offline**, дедлайн не прерывает зависший native
tool, thresholds инженерные. Старые training-label fallbacks остаются. Старые
checkpoints требуют явной декларации preprocessing/units/target/classes в ModelSpec;
неизвестный training provenance автоматически не восстанавливается. Нет физических
устройства/SDK и научной проверки truth/lie. Основной план не помечен завершённым.
Пользовательские исходные правки сохранены; commit/push не выполнялись.

Следующий шаг: продолжать SPEC-01/DATA-02 и проверку реальных labels/протокола;
после подтверждённых данных — причинный processing и полный научный model bundle.
Агентный workflow уже доступен для разрешённых offline записей с явным контрактом.

## Предыдущие задачи и проверки

Выполнен аудит Mendeley TLCG: 39 участников, 8 199 уникальных participant/trial-пар, числовые коды 0/1 у всех участников, дубликатов нет. 454 строки с NA в числовых признаках. Исправлена ошибка прошлого описания: EYES — AOI глаз на изображении, не pupil. Созданы [MENDELEY_AUDIT_RU.md](MENDELEY_AUDIT_RU.md) и scripts/audit_mendeley.py; результаты — outputs/mendeley-audit-20260930-01/audit_summary.json без исходных строк и списка IDs. Код существующих алгоритмов, configs и научные результаты не изменены.

Точные проверки: `.venv/Scripts/python.exe scripts/audit_mendeley.py --output-dir outputs/mendeley-audit-20260930-01` (вне песочницы) — exit code 0; `.venv/Scripts/python.exe -m ruff check scripts/audit_mendeley.py` — All checks passed, exit code 0. Дополнительно: отказ при изменённом хеше источника и согласованность счётчиков прошли; проверены 3 Markdown-файла и 2 локальные ссылки. `git diff --check` — exit code 0. Полный pytest для этого аудит-скрипта не запускался. Новых зависимостей нет. Семантика 0/1 не подтверждена; Supplementary через браузер не получен. Обучение не запускалось; DATA-02 и SPEC-01 не завершены. Сырых временных рядов нет, feedback относится к периоду после ответа.

Начата SPEC-01: пользователь выбрал truth/lie как основной предмет исследования (D-005). Выбор отражён в DECISIONS, PROJECT_CONTEXT и плане. Источник данных, протокол, единица решения и основная метрика ещё не согласованы; SPEC-01 не завершена. Алгоритмы не изменены.


Выполнена ENV-01: проверены `.venv`, Ruff и полный набор тестов, включая LSTM/TCN. Вне песочницы: **22 passed in 32.54s**, exit code 0. Ruff: `All checks passed!`, exit code 0 (в песочнице). Алгоритмы, configs и исходные данные не изменены.

Точные команды успешной проверки из корня (pytest вне песочницы):

```powershell
New-Item -ItemType Directory -Path outputs/env-01-20260930-03
.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp outputs/env-01-20260930-03/pytest-temp
.venv/Scripts/python.exe -m ruff check .
```

Для повторного запуска выбрать новый уникальный родительский каталог: pytest может очищать basetemp. Первый запуск с несуществующим родителем дал 15 passed / 7 setup errors (`FileNotFoundError`). После создания нового родителя запуск в песочнице получил ошибки setup и `PermissionError` при завершении. Вне песочницы с ещё одним новым каталогом все тесты прошли. Ограничение доступа внутри песочницы сохраняется; это не исправление прав доступа.

Ранее подготовлены AGENTS, PROJECT_CONTEXT, WORK_STATUS, DECISIONS, шаблон эксперимента и ссылки README/CONTRIBUTING; создан THESIS_IMPLEMENTATION_PLAN_RU. План остаётся предложением.

## Подтверждено и проверено

- Есть офлайн-обучение, synthetic demo, baseline/LSTM/TCN, тесты и workflow.
- Найдены 16 локальных CIT XLSX. Содержимое и полнота не аудитированы.
- `.venv/Scripts/python.exe` запускается; версии в PROJECT_CONTEXT.
- Для текущих документальных правок `git diff --check` завершился с кодом 0. Проверка через `.venv/Scripts/python.exe` прочитала 7 Markdown-файлов и проверила 24 локальные ссылки: сломанных нет. Полный запуск алгоритмов для этих правок не выполнялся.

В предыдущем аудите системный `python -m pytest -q` дал 15 passed / 7 errors. Ошибки — PermissionError временных папок; повтор с отключённым cacheprovider и новым basetemp тоже получил отказ. Это не установленные дефекты моделей, но suite не подтверждена целиком. Исторические 22 passed / 84% coverage в PROJECT_REPORT не выдавать за свежий результат. Текущий полный запуск через `.venv` вне песочницы прошёл: 22 passed; подробности выше.

## Очередь

`ready` — можно начать при поручении реализации; `needs decision` — зависимая часть ждёт информации; `planned` — не начато. Сама очередь не разрешает внешние действия или тяжёлые эксперименты.

| ID | Статус | Задача и критерий |
|---|---|---|
| ENV-01 | done | Проверить suite через `.venv` и доступные temp каталоги; записать точную команду и результат |
| DATA-01 | ready | Аудит CIT без изменения исходников: manifest, участники, labels, events, frequency, missingness |
| SPEC-01 | needs decision | Truth/lie выбран (D-005); уточнить официальную тему, протокол, единицу решения и основную метрику |
| DATA-02 | needs decision | Открытые агрегаты проверены; подходящих сырых truth/lie-рядов пока нет. Bag-of-Lies отложен (D-006); затем аудит/parser |
| DATA-03 | planned | Строгие labels и сегментация; нет silent defaults, есть fixtures |
| PIPE-01 | planned | Причинность, chunk-invariance, маски разрывов, качество и resets |
| EVAL-01 | planned | Групповые эксперименты, train-only transforms, baseline/LSTM/TCN, uncertainty |
| MODEL-01 | planned | Полный bundle; predict в новом процессе без labels/переобучения |
| STREAM-01 | planned | Replay, ограниченная очередь/память, журнал, измерение задержек |
| LIVE-01 | needs decision | Устройство/SDK; затем интеграция и физический live test |
| UI-01 | planned | Сценарий запуск->наблюдение->отчёт без редактирования кода |
| RELEASE-01 | planned | Чистая установка, воспроизводимые эксперименты и демонстрация |

Следующий шаг: подтвердить codebook 0/1 Mendeley и согласовать отдельный табличный протокол; продолжать поиск сырых truth/lie-рядов. Аудит Mendeley выполнен, потоковый набор он не заменяет. К Bag-of-Lies вернуться после сообщения пользователя (D-006); завершить SPEC-01 по проверенному протоколу. Своих truth/lie-данных и айтрекера у пользователя нет. ENV-01 выполнена. DATA-01 остаётся доступным аудитом отдельного CIT-набора и не подтверждает наличие truth/lie данных.

## Неизвестно

Официальная тема/требования кафедры, срок защиты, возможность будущего доступа к устройству (собственного айтрекера нет), доступ к truth/lie, CPU/GPU, допустимая калибровка и выбранный UI. Не подставлять ответы из предложений.

## Обновление

Заменять описание последней задачи, обновлять статусы по фактическому выполнению. Для проверок указывать команду, среду, результат и ограничения. Оставлять следующий шаг; важные решения связывать с DECISIONS. Не хранить персональные данные, секреты или полный лог.
