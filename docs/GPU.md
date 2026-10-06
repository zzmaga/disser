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
.venv-gpu/Scripts/python.exe manage.py gpu-check --dataset data/processed/text_only_v4_shared --model-name google-bert/bert-base-multilingual-cased --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --head last --fused-adamw --out-file reports/my_gpu_check.json
```

В отчёте проверяются `optimizer_updated`, конечные loss и выделение памяти.
Для проверенного mBERT с fused AdamW пиковая выделенная память составила 2,86 ГБ,
зарезервированная — 3,39 ГБ. Это измерение короткой проверки, не гарантия для любого корпуса.

## Полный запуск

На 4 ГБ используйте microbatch 1, накопление градиента 8, float16,
gradient checkpointing и fused AdamW. Эффективный batch size равен 8.
Путь вывода всегда должен быть новым.

```powershell
.venv-gpu/Scripts/python.exe manage.py train-transformer --dataset data/processed/text_only_v4_shared --out-dir artifacts/my_gpu_mbert_s42 --model-name google-bert/bert-base-multilingual-cased --revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --head last --epochs 2 --batch-size 1 --gradient-accumulation 8 --lr 2e-5 --seed 42 --threads 2 --device cuda --precision float16 --gradient-checkpointing --fused-adamw
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
.venv/Scripts/python.exe manage.py verify-model --run-dir artifacts/my_gpu_mbert_s42 --dataset data/processed/text_only_v4_shared
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
