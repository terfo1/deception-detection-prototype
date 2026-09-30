# Материалы для сдачи задания

Проект: модуль обработки eye-tracking данных и экспериментального моделирования
для Concealed Information Test. Репозиторий:
https://github.com/terfo1/deception-detection-prototype

| Требование | Что показать преподавателю |
|---|---|
| Git | Историю коммитов и тег версии: `git log --oneline --decorate` |
| GitHub | Публичный репозиторий, README, MIT LICENSE, Issues и шаблоны задач |
| CI/CD | Вкладку Actions: тесты на pandas 2/3, проверку LSTM/TCN, сборку пакета и публикацию релиза по тегу |
| Обоснование технологий | Таблицу технологий в README и `docs/PROJECT_REPORT.md` |
| Автотесты | `tests/`, результат pytest и отчёты JUnit/coverage в артефактах Actions |

Демонстрация из корня проекта после активации виртуального окружения:

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not deep'
python scripts/train.py --config configs/synthetic_baseline.yaml
python -m build
git log --oneline --decorate
```

Демо создаёт искусственные данные, извлекает признаки, обучает baseline и
сохраняет метрики и графики в `outputs/synthetic_demo/`. Оно воспроизводимо без
загрузки исходных Excel-файлов. Полученная точность подтверждает выполнение
программы; она не является оценкой качества на реальных участниках.

В рамках подготовки исправлена потеря идентификаторов сегментов на pandas 3 и
порядок нормализации зрачка. Ограничения научной части и дальнейшие задачи
перечислены в `docs/KNOWN_ISSUES.md` и GitHub Issues.
