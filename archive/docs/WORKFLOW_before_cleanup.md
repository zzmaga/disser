# Подготовка данных, обучение и проверка

Актуальная версия — `text_only_v4_shared`, пять стилей и шесть моделей. Она сохраняет
документы и разбиение `text_only_v3`, но согласует фрагменты с двумя токенизаторами.
Старые `pilot_v2` и `text_only_v3` оставлены для истории. Итоги: `reports/PROJECT_STATUS.html`.
Все команды выполняются из корня проекта. Для нового эксперимента задавайте **новые
каталоги вывода**: готовые корпуса и результаты намеренно не перезаписываются.

GPU-окружение и проверенные команды обучения описаны в [GPU.md](GPU.md).
Текущая рукопись: `reports/dissertation_draft_20261006_morphology/manuscript.html`.
Результаты эксперимента с лемматизацией: `reports/morphology_ablation_v1/report.html`.

## Лемматизация и морфологическая обработка

Stanza установлена в отдельном `.venv-gpu`, а не в окружении сайта.
Для воспроизведения: `.venv-gpu/Scripts/python.exe -m pip install -r configs/requirements-morphology.txt`.
Локальные модели загружаются один раз:

```powershell
.venv-gpu/Scripts/python.exe -c "import stanza; stanza.download('kk', model_dir='models/stanza_kk_1_15', package=None, processors={'tokenize':'ktb_nocharlm','mwt':'ktb','pos':'ktb_nocharlm','lemma':'ktb_nocharlm'}, resources_version='1.15.0')"
```

План `configs/experiment_morphology_v1.json` зафиксирован до обработки корпуса.
Последующие шаги работают локально, без передачи текстов стороннему API:

```powershell
.venv-gpu/Scripts/python.exe manage.py annotate-morphology --plan configs/experiment_morphology_v1.json --out-dir data/processed/my_morphology
.venv-gpu/Scripts/python.exe manage.py evaluate-morphology --plan configs/experiment_morphology_v1.json --annotations data/processed/my_morphology --out-dir reports/my_morphology_ablation
```

Сохраняются слова, леммы, POS, морфологические признаки, хеши моделей и входа.
Каждый результат привязан к неизменному sample_id и документу. Пропавшие леммы,
изменённый текст, повторные и отсутствующие записи вызывают ошибку.
Сравниваются raw, stanza_surface и stanza_lemma с LR и словесным SVM. Контроль
stanza_surface отделяет влияние сегментации от дополнительной нормализации.
TF-IDF учится только на train; параметры фиксированы. Основная серия из трёх
seed и выбор модели сайта по этому дополнительному опыту не меняются.

У Stanza предполагаемые, а не экспертные леммы. После опыта сохраняются все шесть
конфигураций и условные парные bootstrap-интервалы, в том числе если улучшения нет.

## Текущие данные

| Этап | Путь | Содержание |
| --- | --- | --- |
| Исходные CSV | `data/*.csv` | Исходники не изменены |
| Внешние источники | `data/external/` | Загрузки, страницы, решения проверки |
| Кандидаты после очистки | `data/cleaned/reviewed_v3/` | 13 408 записей; ещё не экспертная разметка |
| Готовая выборка | `data/processed/text_only_v4_shared/` | 610 текстов, по 122 на каждый из пяти стилей |

В кандидатах: 1000 официальных, 455 научных, 11 600 публицистических,
160 художественных, 193 разговорных записей. После отбора фрагментов, проверки повторов
и группового разбиения в сбалансированную выборку вошло по 74/24/24 документа на стиль
для train/validation/test. Запрос `--per-class 240` задаёт верхнюю границу, а не обещанный размер.

Старые художественные и разговорные записи исключены из нового обучения из-за
ненадёжных меток источников. Они сохранены в карантине, а не удалены.
Из 262 отфильтрованных сообщений ассистент просмотрел все и оставил 193;
это не экспертная разметка. Решения закреплены в `configs/chat_review_20261002.json`.
Восемь образцов заявлений из одного сборника используются только в train.

## Что поступает в модель

`train.csv`, `validation.csv`, `test.csv` имеют ровно два столбца: `text,label`.
Только `text` передаётся классификатору. `metadata.jsonl` хранит ссылки, происхождение,
группы и разбиение для аудита. `manifest.jsonl` связывает текст и метаданные для отчётов;
перед обучением проверяется его точное соответствие CSV и контрольным суммам.

Разная длина строк и переносы внутри корректно заключённого в кавычки поля допустимы.
Загрузчик проверяет структуру CSV и не пропускает повреждённые записи молча.
Очистка удаляет обнаруженные ссылки, HTML, счётчики комментариев, подписи фотографий
и служебные хвосты. Фильтры языка, повторов и таблиц отправляют сомнительное в карантин.
Имена, даты, казахские буквы, пунктуация и эмодзи сохраняются.

Из документа берётся один центральный фрагмент: бюджет 40, 80 или 160 слов,
выбранный детерминированно, и максимум 256 токенов. Все модели получают одинаковый текст.
Это оценка по фрагментам, а не по полному содержанию длинного документа.
Дубликаты и найденные группы близких документов не пересекают части выборки.
У сообщений Telegram нет ID разговоров: группировка блоками строк — приближение.

## Сбор и воспроизведение подготовки

Версии файлов и лицензии закреплены в `configs/external_sources.json`.
Использованы [новости Kurumikz](https://huggingface.co/datasets/kurumikz/kaz-news-corpus)
(ODC-By) и [сборник сообщений Kurumikz](https://huggingface.co/datasets/kurumikz/telegram-corpus-russian-kazakh)
(CC BY-NC-SA 4.0). Условия указаны авторами наборов; исходные карточки сохранены рядом с загрузками.
Художественные тексты собраны из разделов сказок [Bilim-all](https://bilim-all.kz/article/list/4)
и [Ertegiler](https://ertegiler.kz/); заявления — из
[университетского сборника образцов](https://www.nmu.edu.kz/wp-content/uploads/2017/02/ZHOLSILTEME-ANYKTAMA-08.11.2019.pdf).
Разрешение на публичное распространение текстов сайтов не предполагается.

Ниже команды первоначальной сборки. Большинство выходных каталогов уже существует.
Для повторного опыта измените их имена; сохранённую проверку сообщений можно применять
только к тому же снимку кандидатов — команда проверяет его SHA-256.

```powershell
.venv/Scripts/python.exe manage.py download-data --source news --out-dir data/external/news_20261002
.venv/Scripts/python.exe manage.py download-data --source chat --out-dir data/external/chat_20261002
.venv/Scripts/python.exe manage.py import-sources --out-dir data/external/screened_v2_20261002
.venv/Scripts/python.exe manage.py apply-review --candidates data/external/screened_v2_20261002/candidates.jsonl --review configs/chat_review_20261002.json --out-dir data/external/reviewed_20261002
.venv/Scripts/python.exe manage.py collect-data --out-dir data/external/genres_20261002 --per-source 220
.venv/Scripts/python.exe manage.py prepare-data --out-dir data/cleaned/reviewed_v3 --external data/external/reviewed_20261002/candidates.jsonl data/external/genres_20261002/candidates.jsonl
.venv/Scripts/python.exe manage.py build-clean --candidates data/cleaned/reviewed_v3/candidates.jsonl --out-dir data/processed/text_only_v3 --per-class 240 --tokenizer-path data/processed/pilot_v2/tokenizer
```

`collect-data` использует сохранённые страницы и может продолжать сбор. Повторная загрузка
живых сайтов позднее может дать другие данные; для воспроизведения нужны исходные снимки.
Сохранённый tokenizer можно задать через `--tokenizer-path`; без него используется локальный
кеш `kz-transformers/kaz-roberta-conversational`. Данные и веса не входят в Git.

## Обучение и сравнение

```powershell
.venv/Scripts/python.exe manage.py train-classical --dataset data/processed/text_only_v3 --out-dir artifacts/text_only_v3_classical
.venv/Scripts/python.exe manage.py train-transformer --dataset data/processed/text_only_v3 --out-dir artifacts/text_only_v3_roberta_last --head last --epochs 2 --batch-size 8 --threads 4
.venv/Scripts/python.exe manage.py train-transformer --dataset data/processed/text_only_v3 --out-dir artifacts/text_only_v3_roberta_concat4 --head concat4 --epochs 2 --batch-size 8 --threads 4
.venv/Scripts/python.exe manage.py compare --dataset data/processed/text_only_v3 --runs artifacts/text_only_v3_classical artifacts/text_only_v3_roberta_last artifacts/text_only_v3_roberta_concat4 --out-dir reports/text_only_v3
.venv/Scripts/python.exe manage.py diagnose-data --dataset data/processed/text_only_v3 --out-dir reports/text_only_v3_diagnostics
.venv/Scripts/python.exe manage.py evaluate-cohorts --dataset data/processed/text_only_v3 --runs artifacts/text_only_v3_classical artifacts/text_only_v3_roberta_last artifacts/text_only_v3_roberta_concat4 --out-dir reports/text_only_v3_cohorts
```

TF-IDF обучается только на train. Для нейросетей дообучаются encoder и новая голова;
лучшая эпоха выбирается по validation, затем один раз оценивается test.
Настройки двух эпох заданы до сравнения результатов. Dummy — контрольный классификатор,
который не распознаёт стиль; на сайт он не подключён.

Каждый запуск сохраняет веса, параметры, предсказания, метрики по классам, матрицы ошибок,
версии библиотек, SHA-256 данных и снимок исходного кода. `compare` не объединяет разные корпуса.
Отчёт открывается из `reports/text_only_v3/report.html` обычным браузером.

## Сайт и проверки

```powershell
.venv/Scripts/python.exe manage.py serve
.venv/Scripts/python.exe manage.py predict --text-file example.txt --compare
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Сайт: **http://127.0.0.1:8765**. Сервер и CLI используют один inference-сервис,
тот же tokenizer и очистку `text_only_v3`. Длинный ввод ограничивается показанным на сайте
фрагментом. Внешних запросов к ИИ нет. Ответы вычисляются из локальных весов.
Для переключения корпуса используется `configs/deployment.json`.

Тесты проверяют CSV, карантин, отсутствие пересечений групп, соответствие сохранённым
предсказаниям, работу HTTP и неизменность ответа при замене URL издателя в одном тексте.
Примеры пользователя из `configs/diagnostic_examples.json` — известные случаи ошибок,
а не независимый финальный тест. В обучающий набор они специально не добавлялись.

## Граница текущего результата

Это рабочий пилот, **не готовый корпус на 22 600 документов**. Метки требуют экспертной
проверки; разговорные тексты короче остальных, некоторые стили представлены одним основным
источником. Поэтому высокое качество внутреннего test не гарантирует перенос на другие
жанры и сайты. Диагностика отдельно измеряет предсказание по домену и длине — эти данные
основным моделям не передаются. Следующая версия требует дополнительных источников редких
классов и нового зафиксированного внешнего теста; повторное использование текущего test
для подбора решений не будет независимой оценкой.

## Независимая проверка разметки

Готовый пакет: `data/annotation/corpus_audit_20261006/` — 250 текстов, по 50 на
предполагаемый стиль. Это аудит разметки корпуса, **не внешний test**: в пакет могут
входить тексты, использованные при разработке и обучении. Предварительные метки
используются только для отбора. В формах они не показаны.

1. Двум людям, владеющим казахским, передаются отдельно `reviewer_A.html` и
   `reviewer_B.html`. Каждый файл самодостаточен и открывается локально в браузере.
   `private_provenance.jsonl` разметчикам не передаётся.
2. Разметчики читают инструкцию в форме и независимо выбирают метки. Доступны пять
   стилей, смешанный текст, недостаточный контекст, другой язык и мусор. Длинные
   документы ограничены фрагментом; решение относится только к показанному тексту.
3. Каждый указывает собственный постоянный код и скачивает JSON. Можно сохранять
   частичный черновик и восстанавливать его через поле загрузки. Для сравнения
   требуются полные ответы и подтверждение самостоятельной разметки.
4. Полученные файлы сравниваются командой:

```powershell
.venv/Scripts/python.exe manage.py annotate compare --packet data/annotation/corpus_audit_20261006/packet.json --review-a review_R01.json --review-b review_R02.json --out-dir reports/annotation_agreement_01
```

Команда проверяет хеш пакета, соответствие текстов, полноту, уникальность ответов и
различие кодов разметчиков. Она считает согласие и Cohen κ по всем девяти вариантам
решения. Расхождения сохраняются в `needs_adjudication.jsonl`; совпавшие стилевые
решения — в `agreed_styles.jsonl`. Автоматического включения в обучение нет.
Личность, квалификация и фактическая независимость людей не проверяются программой.
Результат после согласования нужно связать с происхождением и разбиением данных.

Пересобрать новый пакет, сохранив прежний:

```powershell
.venv/Scripts/python.exe manage.py annotate build --inputs data/cleaned/reviewed_v3/candidates.jsonl data/external/journals_20261006/candidates.jsonl data/external/long_chat_review_20261006/candidates.jsonl --out-dir data/annotation/next_audit --per-style 50
```

## Дополнительные кандидаты от 06.10.2026

- `data/external/journals_20261006/`: 120 казахских научных аннотаций из трёх
  журналов КазНУ, по 40. Четыре других аннотации помещены в карантин языковым
  фильтром. Сохранены страницы, хеши, происхождение и ссылки на лицензии. Три
  научных документа из уже просмотренной web probe исключены по ID статьи,
  независимо от языка URL. Все новые документы требуют проверки стилевой метки.
- `data/external/long_chat_review_20261006/`: 15 дополнительных сообщений длиной
  40–250 слов после автоматического отбора. В них могут быть цитаты, боты,
  публицистика и другой язык; принадлежность Telegram не является разметкой.
  В ходе просмотра замечен кыргызский пример, пропущенный языковым фильтром.
  Набор предназначен для аудита ошибок фильтра, а не немедленного обучения.

```powershell
.venv/Scripts/python.exe manage.py collect-journals --out-dir data/external/next_journals --per-journal 40 --exclude-probe data/evaluation/web_probe_20261006/samples.json
.venv/Scripts/python.exe manage.py screen-long-chat --source data/external/chat_20261002/telegram_data.txt --existing data/cleaned/reviewed_v3/candidates.jsonl --out-dir data/external/next_chat_review --limit 150
```

Эти операции не меняют `text_only_v3`, активные веса или профиль сайта. Наличие
публичного текста не означает разрешения публиковать весь собранный корпус:
условия использования сохраняются отдельно по источникам.


## Повторная серия с mBERT и общим входом

`configs/experiment_text_only_v4.json` фиксирует seed 42, 43, 44 и две эпохи.
`text_only_v4_shared` сохраняет все 610 документов, метки и разбиения v3. Три
обучающих фрагмента немного укорочены по границам слов для ограничения mBERT.
Validation и test не изменились. Kaz-RoBERTa принимает до 256 своих токенов,
mBERT — до 512 своих токенов; строка текста одинакова у всех классификаторов.

```powershell
.venv/Scripts/python.exe manage.py build-shared --source data/processed/text_only_v3 --out-dir data/processed/text_only_v4_shared --model-source configs/mbert_source.json
.venv/Scripts/python.exe manage.py train-suite --plan configs/experiment_text_only_v4.json --out-dir reports/experiment_text_only_v4
.venv/Scripts/python.exe manage.py summarize-seeds --suite reports/experiment_text_only_v4 --out-dir reports/text_only_v4_repeated
```

Это команды уже запущенной серии; существующие каталоги не перезаписываются.
Для следующего опыта скопируйте план, задайте новый `run_prefix` (например,
`experiment_v5`), новый путь корпуса и его реальный `manifest_sha256`. Новые
каталоги нужны и для серии, и для агрегированного отчёта. Во время обучения
не изменяйте план, корпус и вычислительный код. `status.json` показывает
завершённые запуски, отдельные `.log` — ход каждого обучения.

У mBERT microbatch 2 и накопление градиентов 4; эффективный batch равен 8,
как у Kaz-RoBERTa. Веса encoder и новой головы дообучаются. Сохранены обе
токенизации и ревизии encoder. TF-IDF по-прежнему обучается только на train.
Итого сравниваются шесть типов классификаторов и dummy-контроль, каждый
на трёх seed. Среднее ± стандартное отклонение вычисляется по всем запускам.
Лучшая эпоха определяется по validation. Для демонстрации заранее выбран
seed 42, а семейство основной модели — по среднему validation Macro-F1.

`summary` дополнительно считает 2000 парных bootstrap-повторов с пересэмплированием
групп внутри классов. Это условные интервалы на внутреннем test. Он уже
просматривался при разработке; интервалы не учитывают смещение источников.

Перед подготовкой профиля выполните `verify-model` для трёх нейросетевых
запусков с seed 42. Проверка восстанавливает веса без сети и воспроизводит
по три сохранённых предсказания каждого стиля. Затем:

```powershell
.venv/Scripts/python.exe manage.py prepare-deployment --suite reports/experiment_text_only_v4 --summary reports/text_only_v4_repeated --out-file configs/deployments/text_only_v4_shared.json
.venv/Scripts/python.exe manage.py serve --deployment configs/deployments/text_only_v4_shared.json
```

Профиль содержит SHA-256 весов. Сервис проверяет их перед первой загрузкой,
а также совместимость корпуса и сохранённой токенизации. Для постоянного
переключения скопируйте готовый профиль в `configs/deployment.json` и перезапустите
только сервер сайта. Процессы обучения останавливать не требуется.

## Разбор известных ошибок

```powershell
.venv/Scripts/python.exe manage.py evaluate-diagnostics --examples configs/diagnostic_examples.json --deployment configs/deployments/text_only_v4_shared.json --out-dir reports/text_only_v4_user_diagnostics
.venv/Scripts/python.exe manage.py audit-linear --dataset data/processed/text_only_v4_shared --run artifacts/text_only_v4_classical_s42 --examples configs/diagnostic_examples.json --out-dir reports/text_only_v4_linear_audit
```

Первая команда проверяет два примера пользователя и шесть заранее заданных
замен имён/дат в заявлении. Вторая разбирает сохранённые линейные коэффициенты
и проверяет точное восстановление оценки класса из вкладов TF-IDF × вес.
Эти оценки не являются вероятностями. Такие примеры не добавляются в train
и не используются для выбора модели. Повторную web probe запускайте с
`--purpose regression`, поскольку прежние ответы уже известны.

## Согласование и экспорт человеческой разметки

После `annotate compare` файл `adjudication_template.json` содержит только
расхождения. Человек, разбирающий их, читает показанные тексты, указывает свой
код, заполняет метку и причину каждого решения и подтверждает чтение. Файл
связан с хешами пакета и обоих ответов; перестановка ответов меняет порядок хешей.

```powershell
.venv/Scripts/python.exe manage.py annotate finalize --packet data/annotation/corpus_audit_20261006/packet.json --review-a review_R01.json --review-b review_R02.json --adjudication adjudication_completed.json --out-dir data/annotation/resolved_audit_01
```

Если разногласий нет, `--adjudication` не нужен. Экспорт содержит только
прочитанные фрагменты; метка не распространяется на невидимую часть документа.
Сохраняются ID исходного документа и родительская группа, чтобы не разнести
фрагменты одного источника между train и test. Проверяются контрольные суммы
исходных кандидатов. `reviewed_candidates.jsonl` и `excluded.jsonl` создаются
отдельно: действующий корпус и веса не меняются. Финальное разбиение нужно
зафиксировать отдельно. Программа проверяет структуру ответов, но не удостоверяет
личность и квалификацию разметчиков. Реальные ответы для текущего пакета ещё не получены.


## Новые разговорные и художественные источники

`data/external/forum_questions_clean_20261006/` содержит 96 вопросов форума
Surak/Baribar после удаления повторов и исправления извлечения заголовков и
HTML-комментариев. Один вопрос — один кандидат; ответы и меню не включаются.
Сохраняются ID ветки и, при наличии, хеш ID автора. По длине это всё ещё
преимущественно короткие тексты: медиана 27 слов. Домашние задания, цитаты и
реклама могут проходить автоматический фильтр, поэтому нужна проверка меток.

`data/external/prose_clean_20261006/` содержит 60 публикаций Adebiportal из
разделов «ӘҢГІМЕ» (ID 51) и «ПРОЗА» (ID 29). Ещё 30 публикаций раздела ID 41
выведены в карантин: при таком же названии раздел смешивает прозу, критику,
мемуары и публицистику. Медиана оставшихся публикаций — 2966 слов. Подсчёт
ведётся по публикациям, а не по нарезанным абзацам. Перед разбиением следует
проверить авторов, части одного произведения и повторы. Снимки всех 60 страниц
проверены по SHA-256; новые метки остаются предварительными.

```powershell
.venv/Scripts/python.exe manage.py collect-forum --out-dir data/external/next_forum --per-category 25 --max-pages 8
.venv/Scripts/python.exe manage.py collect-prose --out-dir data/external/next_prose --per-category 30 --max-pages 8 --exclude-probe data/evaluation/web_probe_20261006/samples.json
```

Сбор ограничен по страницам, частоте запросов и объёму ответа; учитываются
robots.txt и ответы ограничения доступа. URL сохраняются в происхождении,
а из текста удаляются. Условия источников сохранены; открытая публикация
страницы не обозначается как свободная лицензия на весь корпус.
