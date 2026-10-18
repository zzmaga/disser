# Обучение на видеокарте

На этом компьютере проверена NVIDIA GeForce RTX 3050 Laptop GPU с 4 ГБ памяти.
Основное окружение `.venv` остаётся CPU-окружением сайта и завершённого сравнения.
Для дальнейшего обучения создано отдельное `.venv-gpu`: Python 3.14,
PyTorch `2.14.0+cu126`, CUDA runtime 12.6. Драйвер — 566.07.

## Установка и проверка

Окружение уже установлено. Для воспроизведения на совместимом компьютере:

```powershell
python -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r configs/requirements-gpu.txt
.venv-gpu/Scripts/python.exe -m pip check
.venv-gpu/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Сначала запускайте короткую проверку памяти. Она использует только один из самых
длинных **train**-фрагментов, делает три обновления и не сохраняет веса:

```powershell
.venv-gpu/Scripts/python.exe manage.py gpu-check --dataset data/processed/text_only_v4_shared --model-name google-bert/bert-base-multilingual-cased --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --head last --fused-adamw --eligible-only --out-file reports/my_gpu_check.json
```

В отчёте проверяются `optimizer_updated`, конечные loss и выделение памяти.
Для проверенного mBERT с fused AdamW пиковая выделенная память составила 2,86 ГБ,
зарезервированная — 3,39 ГБ. Это измерение короткой проверки, не гарантия для любого корпуса.

## Полный запуск

### Проверенный новый encoder: mmBERT-small

В ходе продолжения 18–19 октября mmBERT-small закреплён на ревизии
`abc32620dd4f6ab06f5fbe905dc25f310618e09f`. Три discarded train-only шага
прошли: 140 495 621 параметр с пяти-классовой головкой, 256 токенов,
allocated 2,115 GiB, reserved 2,211 GiB; 3 успешных обновления, без OOM.
Использован штатный ModernBERT из Transformers 5.17.0; FlashAttention не установлен
специально. Это измерение одного входа, не замена полной проверке обучения.
Доказательство: `reports/technical/dissertation_completion_20261018/encoder_compatibility/mmbert_gpu.json`.

Уже выполненную проверку не повторять ради числа. Для воспроизведения на новом
компьютере с новым именем результата:

```powershell
.venv-gpu/Scripts/python.exe manage.py gpu-check --dataset data/processed/expanded_v6_shared --model-name jhu-clsp/mmBERT-small --revision abc32620dd4f6ab06f5fbe905dc25f310618e09f --head last --fused-adamw --tokenizer-reference data/processed/tokenizer_reference_v7_20261018 --eligible-only --fit-reference-view --allow-download --out-file reports/technical/NEW_MMBERT_GPU_CHECK.json
```

`--eligible-only` удаляет резервные строки из старого **train** до выбора входа.
`--fit-reference-view` явно удаляет хвостовые целые слова до ограничений всех
трёх токенизаторов только для технического probe. Исходный dataset не меняется;
его прежние `sample_id` и хешы показанного probe-входа записываются отдельно.
Первый preflight без этой проекции остановился до загрузки модели из-за длины.
Веса трёх шагов отброшены, validation/test не оценивались.

### Настройки основного обучения

На 4 ГБ используйте microbatch 1, накопление градиента 8, float16,
gradient checkpointing и fused AdamW. Эффективный batch size равен 8.
Путь вывода всегда должен быть новым.
В примере заменить `NEW_FROZEN_UNRESERVED_DATASET` на новую зафиксированную
выборку после проверки меток и резерва. Старые v4/v6 содержат теперь
зарезервированные source units: новый trainer правомерно отклоняет их повторное
обучение; проверка сохранённых исторических весов продолжает работать.

```powershell
.venv-gpu/Scripts/python.exe manage.py train-transformer --dataset data/processed/NEW_FROZEN_UNRESERVED_DATASET --out-dir artifacts/my_gpu_mbert_s42 --model-name google-bert/bert-base-multilingual-cased --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --head last --epochs 2 --batch-size 1 --gradient-accumulation 8 --lr 2e-5 --seed 42 --threads 2 --device cuda --precision float16 --gradient-checkpointing --fused-adamw
```

Для Kaz-RoBERTa замените имя на `kz-transformers/kaz-roberta-conversational`,
ревизию на `43077c2fd0a163487ed468b5ec3b8750686a5888` и задайте новый `--out-dir`.
`--head last` использует последний слой, `--head concat4` — четыре последних.
Предобученные веса должны находиться в локальном кеше; для первой загрузки
можно явно добавить `--allow-download`.

Код сохраняет `optimizer_steps` и `skipped_optimizer_steps` в `history.json`.
GradScaler начинает с масштаба 128; scheduler не продвигается при пропущенном
обновлении. Проверка validation/test выполняется в float32, веса сохраняются
в float32 и подходят для CPU-сайта.

После обучения проверьте восстановление без сети:

```powershell
.venv/Scripts/python.exe manage.py verify-model --run-dir artifacts/my_gpu_mbert_s42 --dataset data/processed/NEW_FROZEN_UNRESERVED_DATASET
```

06.10.2026 выполнены полные проверочные эпохи mBERT и Kaz-RoBERTa concat4:
по 370 train-текстов, 47 успешных обновлений, без пропусков. Артефакты находятся
в `archive/artifacts/gpu_pipeline_check_mbert_s42/` и
`archive/artifacts/gpu_pipeline_check_kaz_concat4_s42/`. Это проверка новой реализации;
её результаты не добавляются к основной серии CPU из трёх seed и не меняют выбор
модели сайта. Сравнивать время разных режимов как строгий benchmark нельзя.

Расширенная серия v6 также завершена: 9 нейросетевых запусков по две эпохи,
по 246 успешных обновлений без пропусков, 980 train-текстов. Дополнительно завершены
source-holdout (100 обновлений) и проверка длины train (246). Все 11 сохранённых
нейросетевых запусков проверены на CPU. Истории и подтверждения находятся вместе
с весами в `archive/artifacts/`; сводка —
`reports/technical/expanded_final_training_history.json`.

## Рисунки для рукописи

```powershell
.venv-gpu/Scripts/python.exe -m pip install -r configs/requirements-figures.txt
.venv-gpu/Scripts/python.exe manage.py export-figures --summary reports/expanded_v6/repeated --out-dir reports/my_figures
```

Экспортируются PNG и SVG, каждый рисунок строится из сохранённых результатов.

## Документация

- [PyTorch: CUDA-сборки](https://pytorch.org/get-started/previous-versions/).
- [Automatic mixed precision и GradScaler](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).
- [AdamW и fused](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
- [Transformers: gradient checkpointing](https://huggingface.co/docs/transformers/main_classes/model).
