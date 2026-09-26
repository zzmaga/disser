# KazStyle — определение стиля казахского текста

Локальный прототип для трёх стилей: **официально-деловой, публицистический и художественный**.
Вставьте текст на сайте и получите ответ модели или сравнение пяти классификаторов.
Внешний ИИ API не используется.

## Запуск сайта

В существующем окружении дважды нажмите `start_site.cmd` или выполните из корня проекта:

```powershell
.venv/Scripts/python.exe manage.py serve
```

Откройте **http://127.0.0.1:8765**. Окно сервера должно оставаться открытым.

После клонирования нужны Python, зависимости из `requirements.txt` и локальные данные/веса.
Каталоги `data/`, `artifacts/`, `reports/`, `.venv/` не включены в Git. Для готового сайта нужны
`data/processed/pilot_v2/` и три папки запусков: `artifacts/pilot_v2_classical/`,
`artifacts/pilot_v2_roberta_last/`, `artifacts/pilot_v2_roberta_concat4/`.
Без них сначала подготовьте корпус и обучите модели по [инструкции](docs/WORKFLOW.md).
Версии текущего окружения: [configs/requirements-installed.txt](configs/requirements-installed.txt).

## Что уже обучено

- TF-IDF + Logistic Regression, словесный TF-IDF + SVM, символьный TF-IDF + SVM.
- Два варианта Kaz-RoBERTa: последний слой и объединение четырёх последних слоёв.
  Взята предобученная `kz-transformers/kaz-roberta-conversational`; encoder и новая
  классификационная голова дообучены локально по две эпохи.
- Единый корпус: **630 train, 135 validation, 135 test**, поровну на три класса.

Высокие метрики относятся к небольшому пилоту. Каждый класс связан с одним сайтом,
поэтому качество на новых источниках ещё требует проверки. Это не подтверждение цифр статьи.

## Структура

```text
kazstyle/     сервер, обработка данных, обучение, модели и оценка
frontend/     интерфейс сайта
configs/      версии окружения
data/         исходные CSV, подготовленные корпуса и внешние кандидаты
artifacts/    обученные модели и параметры запусков
reports/      сгенерированные отчёты
docs/         инструкции и материалы исследования
tests/        автоматические проверки
archive/      старые сборщики, эксперименты и временные материалы
manage.py     единая точка запуска команд
start_site.cmd
```

Подробно: [архитектура](docs/ARCHITECTURE.md), [подготовка данных и обучение](docs/WORKFLOW.md),
[результаты пилота](docs/research/2026-09-25/PILOT_RESULTS_RU.md).
Старые записки могут ссылаться на прежние имена файлов; актуальные команды находятся в инструкции.

```powershell
.venv/Scripts/python.exe manage.py --help
.venv/Scripts/python.exe manage.py train-transformer --help
.venv/Scripts/python.exe manage.py predict --text "Қазақстан туралы мәтін" --compare
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Исходные CSV, сохранённые веса и результаты обучения при реорганизации не изменялись.
Исторические `source_snapshot` и provenance внутри запусков сохраняют код и пути на момент обучения.
