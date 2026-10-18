"""Render one current research hub from the activated, verified deployment."""
import html
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from kazstyle.data.corpus import load_manifest,file_hash,write_json
from kazstyle.inference.deployment import FAMILIES
from kazstyle.settings import PROJECT_ROOT,evidence_path


def dissertation_plan_html():
    """Dated research roadmap; detailed decisions live in docs/PROJECT.md."""
    return '''<section id="dissertation-plan"><h2>План завершения диссертации — сверка 18 октября</h2>
<p>Диссертация сдаётся на английском. Тема — построение корпуса и классификация пяти функциональных стилей казахского текста. Прототип и пилотное сравнение работают. Пользователь поручил содержательную проверку ассистенту: такие решения сохраняются как AI-assisted curation, без заявлений о независимой экспертной разметке.</p>
<p>Рабочая цель пользователя — подготовить работу за месяц, до 11 ноября 2026. В предоставленном ИПМ срок представления — июнь 2027, защита в календарном плане — апрель–май 2027. Это разные сроки; ближайшая предзащита и действующие требования допуска ещё не подтверждены.</p>
<table><tr><th>Ориентир</th><th>Работа</th><th>Условие завершения</th></tr>
<tr><td>18–22 октября</td><td>Исправление потерь слов, разбор жанров и спорных меток; резервирование групп до обучения</td><td>Версия корпуса с хешами и происхождением решений; отдельно неизвестный источник и диагностические тексты. Недобор показывается явно, квота не заполняется повторами</td></tr>
<tr><td>23–29 октября</td><td>Фиксация выборки; один технический запуск и итоговая серия</td><td>Три линейных варианта, mBERT, Kaz-RoBERTa last, кандидат mmBERT-small после проверки совместимости; три seed, общий вход, выбор по validation</td></tr>
<tr><td>30 октября – 5 ноября</td><td>Оценка на зарезервированных группах, анализ ошибок, завершение английских глав</td><td>Происхождение меток явно указано. AI-проверка не называется человеческим gold; source-holdout, метрики классов и интервалы; test не используется для подбора</td></tr>
<tr><td>6–11 ноября</td><td>Оформление, воспроизведение, демонстрация и доклад</td><td>Согласованный шаблон, паспорт корпуса, инструкция запуска, проверяемые таблицы; резерв на замечания</td></tr></table>
<p class="note">Даты — рабочий ориентир, не подтверждённый срок допуска. Неизвестная при обучении выборка с AI-метками всё равно не доказывает независимость разметки. Предлагаемый протокол: <code>configs/research_protocol_v7.json</code>; он пока не является планом запуска. Новые веса ещё не обучались.</p>
<h2>Что статья и старый план пока не подтверждают</h2>
<table><tr><th>Утверждение</th><th>Результат сверки</th></tr>
<tr><td>22 600 документов, 6,3 млн токенов, LR Macro-F1 0,76</td><td>Исходный эксперимент не подтверждён сохранёнными артефактами. Новый банк и новая серия оформляются отдельно; слова и токены encoder не смешиваются.</td></tr>
<tr><td>Ручная проверка 500 документов и коэффициенты κ</td><td>Подтверждающих исходных ответов нет. Нужна новая реально выполненная разметка; ответы модели и метки сайта её не заменяют.</td></tr>
<tr><td>KazBERT и конкатенация четырёх слоёв</td><td>В статье обучение предложено как следующий этап. Сейчас проверены mBERT и Kaz-RoBERTa last/concat4; concat4 не превзошёл last по среднему validation v6. Названия моделей нельзя подменять.</td></tr>
<tr><td>«Первый» корпус, ссылки и методика очистки</td><td>Национальный корпус уже описывает пять стилей. Ссылка [9] arXiv:2011.13175 относится к оптоволоконному датчику. В текущем очистителе Lingua, не заявленный в статье fastText-фильтр. Обзор и методику нужно писать по проверенным источникам и коду.</td></tr>
<tr><td>115 000 сегментов, GAN/style transfer, XLM-R, Streamlit</td><td>Это будущие направления или предложения старого плана. Они не добавляются автоматически к обязательной реализации. Работа посвящена корпусу и классификации пяти функциональных стилей.</td></tr></table>
<p>Проверенные первоисточники: <a href="https://qazcorpus.kz/header/belgileren.php">описание Национального корпуса</a>, <a href="https://arxiv.org/abs/2011.13175">фактическая работа по ссылке [9]</a>, <a href="https://huggingface.co/kz-transformers/kaz-roberta-conversational">карточка используемого encoder</a>.</p>
<p>ИПМ содержит план трёх публикаций; это не подтверждение действующего требования допуска или обязательности Scopus. Найденная публичная методичка AITU допускает публикацию и/или выступление на конференции, но актуальность для набора пользователя не подтверждена. Нужны действующие требования кафедры, доказательство статуса статьи и шаблон. <a href="../docs/PROJECT.md">Сверка и решения</a>.</p></section>'''


def build():
    root=PROJECT_ROOT;reports=root/'reports';technical=reports/'technical'
    profile=json.loads((root/'configs/deployment.json').read_text(encoding='utf-8'))
    active_frame,active_config=load_manifest(root/profile['dataset'])
    active_summary=evidence_path(profile.get('evidence',{}).get('summary','reports/text_only_v4_repeated'))
    if file_hash(active_summary/'results.json')!=profile['summary_sha256']:raise ValueError('Active deployment summary changed')
    summary=evidence_path('reports/expanded_v6/repeated')
    frame,config=load_manifest(root/'data/processed/expanded_v6_shared')
    provenance=json.loads((summary/'provenance.json').read_text(encoding='utf-8'))
    if provenance['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Research summary dataset differs')
    results=json.loads((summary/'results.json').read_text(encoding='utf-8'))
    rows=''.join('<tr><td>'+html.escape(name)+'</td><td>'+f"{results[family]['validation_macro_f1_mean']:.4f} ± {results[family]['validation_macro_f1_sample_std']:.4f}"+'</td><td>'+f"{results[family]['test_macro_f1_mean']:.4f} ± {results[family]['test_macro_f1_sample_std']:.4f}"+'</td></tr>' for family,(_,name,_) in FAMILIES.items())
    def link(relative,title):
        target=evidence_path(relative)
        if not target.exists():raise FileNotFoundError(target)
        return '<a href="'+html.escape(Path(os.path.relpath(target,reports)).as_posix(),quote=True)+'">'+html.escape(title)+'</a>'
    links=[('reports/corpus_expansion_20261006/report.html','Корпус: очистка, источники и проверки'),
           (str(summary/'report.html') if not summary.is_absolute() else str(summary.relative_to(root)/'report.html'),'Основное сравнение: все seed и интервалы'),
           ('reports/expanded_v6/source_holdout/report.html','Перенос на другие источники'),
           ('reports/expanded_v6/length_ablation/report.html','Проверка коротких обучающих фрагментов'),
           ('reports/expanded_v6/user_diagnostics/report.html','Заявление, фотосинтез и изменения имён/дат'),
           ('reports/expanded_v6/web_probe/report.html','Повторная проверка 14 известных веб-текстов'),
           ('reports/dissertation_english/manuscript.html','English working manuscript — актуальная основа диссертации'),
           ('reports/dissertation_expanded_20261006/manuscript.html','Предыдущая русская рабочая рукопись'),
           ('reports/ipm_progress_20261006.html','Восемь задач ИПМ: сделано и осталось'),
           ('reports/text_only_v4_repeated/report.html','Стабильная версия v4: предыдущая серия'),
           ('reports/morphology_ablation_v1/report.html','Отдельное сравнение лемматизации на v4')]
    audit_note = ''
    audit_path = reports/'corpus_audit_20261010/summary.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        links[:0] = [('reports/corpus_audit_20261010/corpus.html','Просмотреть корпус v6: тексты, стили и поиск'),
                     ('reports/corpus_audit_20261010/report.html','Аудит 10 октября: мусор, размер корпуса и высокий F1')]
        audit_note = ('<h2>Предыдущий аудит пилотного корпуса</h2><p>До этой ревизии и отбора: '
            + str(audit['merged_candidates']) + ' кандидата; в рабочем пуле: '
            + str(audit['working_pool']) + '. Эти числа не складываются с размером v6. '
            + 'Диагностика только по источнику, без текста: Macro-F1 '
            + f"{audit['domain_baseline']['test_macro_f1']:.4f}"
            + '. Это сильная связь источника с классом, а не доказательство качества на новых текстах.</p>'
            + '<p>Отдельная ревизия очистки изменила '
            + str(audit['candidate_revision']['changed']) + ' кандидатов; '
            + str(audit['candidate_revision']['review_required']) + ' записей отмечены для просмотра. '
            + 'Старые выборки и веса сохранены; нового обучения не было.</p>')
    corpus_note = ''
    corpus = None
    registry_path = root/'configs/corpus_registry.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {}
    corpus_relative = registry.get('active_document_corpus', 'data/cleaned/unified_corpus_20261010')
    corpus_path = root/corpus_relative
    if (corpus_path/'COMPLETE.json').exists():
        checksums = json.loads((corpus_path/'COMPLETE.json').read_text(encoding='utf-8'))
        if file_hash(corpus_path/'summary.json') != checksums['summary.json']:
            raise ValueError('Unified corpus summary checksum mismatch')
        corpus = json.loads((corpus_path/'summary.json').read_text(encoding='utf-8'))
        links[:0] = [(corpus_relative+'/corpus.html','Единый корпус: состав и CSV'),
                     (corpus_relative+'/corpus.csv','Скачать весь корпус — text,label')]
        corpus_note = ('<h2>Новый единый корпус — обучение ещё не выполнено</h2><p>'
            + str(corpus['accepted_provisional']) + ' документов в основном CSV; '
            + str(corpus['review']) + ' на разбор; '+str(corpus['excluded'])+' исключено. '
            + 'Метки предварительные. Длинные тексты не нарезаны ради увеличения числа документов.</p>'
            + '<p>Лимит 1000/стиль снят. Ссылки и сведения об источниках хранятся отдельно от модельного текста. '
            + 'Эта версия не заменяет корпус и метрики завершённого v6: сайт пока использует прежние веса.</p>')
        current_targets = registry.get('collection_targets') or corpus.get('collection_targets', {})
        corpus_rows = ''.join(f'<tr><td>{html.escape(style)}</td><td>{count:,}</td>'
                             f'<td>{current_targets.get(style, 2000):,}</td></tr>'
                             for style, count in corpus['by_style'].items())
        corpus_note += ('<p>Текущий банк: <code>'+html.escape(corpus_relative)+'</code>. '
                       'Цель относится к самостоятельным документам, не к фрагментам. Публицистика не добирается.</p>'
                       '<table><tr><th>Стиль</th><th>Принято предварительно</th><th>Ориентир</th></tr>'
                       +corpus_rows+'</table>')
    lengths_dir = root/registry.get('length_audit', 'reports/technical/corpus_scale_20261012/lengths_expanded')
    if not registry.get('length_audit') and not (lengths_dir/'summary.json').exists():
        lengths_dir = reports/'technical/corpus_scale_20261012/lengths'
    lengths_path = lengths_dir/'summary.json'
    if registry.get('length_audit') and not lengths_path.exists():
        raise FileNotFoundError('Registered length audit is missing: '+str(lengths_path))
    if lengths_path.exists():
        links.append(((lengths_dir/'report.html').relative_to(root).as_posix(),
                      'Длины документов, структура CSV и доля текста во входе модели'))
        length_check = json.loads(lengths_path.read_text(encoding='utf-8'))
        if (root/length_check['corpus']).resolve() != corpus_path.resolve():
            raise ValueError('Length audit belongs to another corpus; update the registry')
        length_rows = ''.join(f'<tr><td>{html.escape(style)}</td><td>{v["words"]["median"]:g}</td>'
                              f'<td>{v["words"]["max"]:,}</td><td>{v["excerpt_words"]["median"]:g}</td>'
                              f'<td>{100*v["word_coverage"]["median"]:.1f}%</td></tr>'
                              for style, v in length_check['by_style'].items())
        structure_ok = all(item['valid_structure'] for item in length_check['csv_structure'])
        corpus_note += ('<h2>Почему длины важны</h2><p>Проверено документов: '
                       +str(length_check['documents'])+'. Структура CSV: '
                       +('корректна' if structure_ok else 'есть ошибки')+'. Банк: '
                       +html.escape(length_check['corpus'])+'. Ниже — исходные документы этого банка '
                       'при обработке текущим алгоритмом выбора фрагмента v4; это не новое обучение.</p>'
                       '<table><tr><th>Стиль</th><th>Медиана слов</th><th>Максимум слов</th>'
                       '<th>Медиана слов во входе</th><th>Медиана доли прочитанных слов</th></tr>'
                       +length_rows+'</table><p>Один длинный документ остаётся одним документом. '
                       'Переносы строк внутри поля CSV не создают новые документы. '
                       'Большие приложения-таблицы и ошибочные стилевые метки требуют содержательного разбора.</p>')
        prose = next((item for item in length_check.get('by_style_and_genre', [])
                      if item['genre'] == 'literary_prose_candidate'), None)
        if prose:
            corpus_note += ('<p><strong>Общая медиана художественного класса скрывает разницу жанров.</strong> '
                f'Для прозы ({prose["documents"]} документов) медиана — {prose["words"]["median"]:g} слов, '
                f'медианная доля входа — {100*prose["word_coverage"]["median"]:.1f}%. '
                'Короткие стихи не устраняют ограничение обработки длинной прозы.</p>')
    project_audit_note = ''
    project_audit = reports/'technical/project_audit_20261012/audit.json'
    if project_audit.exists():
        check = json.loads(project_audit.read_text(encoding='utf-8'))
        names = {'official':'Официально-деловой', 'scientific':'Научный', 'publicistic':'Публицистический',
                 'literary':'Художественный', 'colloquial':'Разговорный'}
        audit_rows = ''.join(
            f'<tr><td>{names[style]}</td><td>{item["documents"]:,}</td><td>{item["target"]:,}</td>'
            f'<td>{item["shortfall"]:,}</td><td>{item["words_p10_median_p90"][1]:,}</td></tr>'
            for style, item in check['bank_by_style'].items())
        project_audit_note = ('<h2>Состояние до расширения — первый аудит 12 октября 2026</h2>'
            '<p>Корпус и сохранённые источники прошли проверку контрольных сумм. '
            'Основной корпус и новые партии учитываются отдельно:</p>'
            '<table><tr><th>Стиль</th><th>В корпусе</th><th>Цель</th><th>До цели</th><th>Медиана слов</th></tr>'
            + audit_rows + '</table><p>Отдельно сохранено <strong>'
            + f'{check["candidate_rows_not_added_to_bank"]:,}' + ' кандидатов</strong>. Часть сбора прервалась; '
            'на момент первого аудита эти строки ещё не прошли объединённый отбор. '
            'Текущие числа приведены в разделе нового корпуса выше.</p>'
            '<p class="note">Основные пробелы: преобладание законов среди деловых текстов и аннотаций среди научных, '
            'различие длин, зависимость стиля от источника, предварительное происхождение меток. '
            'Равное число строк само по себе эти проблемы не устраняет.</p>'
            '<p>Исправлены дедупликация форумных реплик, сохранение связей копий при группировке, '
            'обработка HTTP-запросов и лишняя зависимость отбора данных от Transformers. '
            'Старые данные и веса не переписаны. '
            '<a href="technical/project_audit_20261012/audit.json">Числа и доказательства аудита</a>; '
            '<a href="../docs/PROJECT.md">подробности и оставшиеся задачи</a>.</p>')
    # Write the IPM page first, so every hub link is checked against a real file.
    req=json.loads((reports/'dissertation_audit_20261006/requirements.json').read_text(encoding='utf-8'))['requirements']
    updates=[
        ('В работе','18 октября сверены исходные ИПМ и статья; подготовлена английская рабочая рукопись.','Дописать обзор по полным первоисточникам, уточнить новизну и ссылки; классификация стиля остаётся основной задачей.'),
        ('Рабочий корпус построен','13 973 кандидата после объединения; v6 содержит 1 265 сбалансированных текстовых единиц из 15 источников.','Расширить проверенные жанры; размер сам по себе не подтверждает репрезентативность.'),
        ('AI-assisted curation, не экспертный gold','888 полных разговорных кандидатов прочитаны, 810 включены в новую версию. Отдельно прочитаны 50 точных резервных views: 43 предварительно сохранены, 7 требуют разбора. Аудиты могут пересекаться; ответов людей нет.','Проверить остальные резервные views по замороженной инструкции; хранить причины и объём просмотра. Не вычислять человеческое согласие по ответам ИИ.'),
        ('Реализовано и проверено','Очистка, три закреплённых токенизатора для будущей серии, полные авторские/текстовые группы и защита резерва. Лемматизация Stanza исследована отдельно на пилоте v4.','Завершить аудит точных views и преобразований перед созданием итоговой выборки.'),
        ('Пилотное сравнение завершено','Шесть вариантов, seed 42/43/44; все 12 заданий v6 завершены, веса восстановлены на CPU. Совместимость mmBERT-small с GPU подтверждена тремя discarded train-only шагами.','Одна новая сравнительная серия после фиксации точных входов и меток; короткий probe не является готовой моделью или новым показателем точности.'),
        ('Работает','Локальный сайт и сервер, пять стилей, шесть моделей; общий фрагмент и отметка частичного анализа; внешний ИИ API не используется.','Проверить восстановление полного сайта из чистой среды, зафиксировать демонстрационный сценарий и ограничения.'),
        ('Внутренняя оценка выполнена; новый резерв защищён','Метрики, три seed, групповые интервалы, source-holdout и известные примеры. Для будущей v7 зарезервированы 1 000 кандидатов; 997 views, 0 флагов против записанных v4/v6 по заданным порогам.','Закончить аудит и метки, затем оценить новые модели. Пока нет результатов по резерву или независимого человеческого gold.'),
        ('Частично','PDF статьи сохранён; о принятии сообщил пользователь. Язык диссертации — английский; есть рабочий текст.','Подтвердить статус статьи и действующие условия допуска, получить шаблон; завершить текст и доклад. Старые неподтверждённые числа не выдавать за новый эксперимент.')]
    if corpus:
        updates[1] = ('Новая версия корпуса подготовлена',
                      str(corpus['accepted_provisional'])+' документов в едином CSV, источники и группы отдельно; метки предварительные.',
                      'Новый ориентир 11 425/стиль; 2 000 — промежуточный этап. Не все кандидаты приняты: см. реестр партий. Нужны разнообразие жанров, разметка и фиксированные входы для обучения.')
    reservation_dir=registry.get('evaluation_reservation')
    if reservation_dir:
        from kazstyle.data.reservation import registered_ledgers
        ledgers=registered_ledgers()
        reservation=json.loads((root/reservation_dir/'summary.json').read_text(encoding='utf-8'))
        frozen=json.loads((root/reservation_dir/'COMPLETE.json').read_text(encoding='utf-8'))
        if file_hash(root/reservation_dir/'summary.json')!=frozen['summary.json']:
            raise ValueError('Reservation summary changed')
        if not any(ledger['reservation_id']==reservation['reservation_id'] for _,ledger in ledgers):
            raise ValueError('Reservation is not registered for training protection')
        if registry.get('reservation_checks'):
            combined=json.loads((root/registry['reservation_checks']).read_text(encoding='utf-8'))
            for path,sha in combined['ledgers'].items():
                if file_hash(root/path)!=sha:raise ValueError('Combined reservation check is stale')
            reservation={**reservation,**combined}
        reserve_rows=''.join(f'<tr><td>{html.escape(style)}</td><td>{count:,}</td>'
            f'<td>{reservation["embargo_by_style"][style]:,}</td><td>{reservation["eligible_by_style"][style]:,}</td></tr>'
            for style,count in corpus['by_style'].items())
        corpus_note+=('<h2>Резерв до следующего обучения</h2><p><strong>'
            +str(reservation['candidates'])+' кандидатов</strong>: по 100 каждого стиля для проверки новых групп '
            'и по 100 для проверки источников. Полные связанные группы и выбранные источники исключены '
            'из нового обучения и validation: '+str(reservation['embargo_documents'])+' документов.</p>'
            '<table><tr><th>Стиль</th><th>В банке</th><th>Вне обучения</th><th>Доступно для разработки</th></tr>'
            +reserve_rows+'</table><p class="note">Это резерв с предварительными метками, ещё не готовый итоговый test. '
            'Некоторые источники знакомы историческим моделям: исключение действует для будущей серии v7. '
            'Журнал, портал и платформа отзывов не равны независимым издателям. '
            'Оригинальные тексты корпуса сохранены; источник не подаётся классификатору.</p>')
        links.append((reservation_dir+'/summary.json','Резерв: размеры, роли и исключение из обучения'))
        if registry.get('reservation_checks'):
            links.append((registry['reservation_checks'],'Общие проверки резерва, фрагментов и защищённых групп'))
        if registry.get('evaluation_views'):
            views=json.loads((root/registry['evaluation_views']/'summary.json').read_text(encoding='utf-8'))
            corpus_note+='<p>Подготовлено '+str(views['views'])+' точных резервных фрагментов под три токенизатора; '+str(views['quarantined'])+' на отдельный разбор. Семантическую проверку меток нужно завершить.</p>'
            links.append((registry['evaluation_views']+'/summary.json','Точные фрагменты резерва и карантин'))
        if registry.get('development_views'):
            development=json.loads((root/registry['development_views']/'summary.json').read_text(encoding='utf-8'))
            corpus_note+='<p>Доступно '+str(development['views'])+' общих модельных фрагментов для подготовки train/validation. После аудита трёх пар общих фрагментов дополнительно исключены 6 записей полных групп. Исходные тексты сохранены, новых метрик нет.</p>'
            links.append((registry['development_views']+'/summary.json','Актуальные модельные фрагменты без резервных групп'))
        probe_path=registry.get('encoder_compatibility')
        if probe_path:
            probe=json.loads((root/probe_path).read_text(encoding='utf-8'))
            if probe['status']!='passed' or not all(probe['optimizer_updated']):
                raise ValueError('Registered encoder probe did not pass')
            corpus_note+=('<p><strong>mmBERT-small помещается в GPU 4 GB:</strong> 3/3 обновления, '
                '256 токенов, пиковая allocated память '+f'{probe["peak_allocated_bytes"]/2**30:.2f}'
                +' GiB. Временные веса отброшены; это техническая проверка, не новый Macro-F1.</p>')
            links.append((probe_path,'mmBERT-small: проверка CUDA и памяти'))
    for row,update in zip(req,updates):row[1:]=list(update)
    css='body{font:16px/1.6 system-ui;background:#f3f6fa;color:#183048;margin:0}main{max-width:1160px;margin:30px auto;background:white;padding:34px;border-radius:16px}h1,h2{line-height:1.2}a{color:#175ca8}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #d7e0e8;text-align:left;vertical-align:top}.note{padding:16px;background:#fff2d7}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.cards a{padding:16px;background:#edf4fb;border-radius:10px;text-decoration:none}.muted{color:#5e7081}code{overflow-wrap:anywhere}@media(max-width:650px){main{margin:0;padding:18px}table{display:block;overflow:auto}}'
    ipm_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in req)
    plan_note=dissertation_plan_html()
    (reports/'ipm_progress_20261006.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ИПМ: актуальное состояние</title><style>'+css+'</style><main><p><a href="PROJECT_STATUS.html">К результатам</a></p><h1>Восемь задач ИПМ</h1><p>Обновлено 18 октября 2026. Тема: построение корпуса и системы определения текстового стиля на казахском языке с использованием машинного обучения. Завершённые программные этапы отделены от незавершённой итоговой оценки.</p><table><tr><th>Задача</th><th>Статус</th><th>Что подтверждено</th><th>Что осталось</th></tr>'+ipm_rows+'</table>'+plan_note+'</main></html>',encoding='utf-8')
    if registry.get('current_corpus_report'):
        links.insert(0, (registry['current_corpus_report'], 'Последнее расширение: изменения, проверки и следующий шаг'))
    if registry.get('curation_report'):
        links.insert(0, (registry['curation_report'], '888 полных текстов: решения проверки и причины исключений'))
    cards=''.join(link(path,title) for path,title in links)
    hold=json.loads(evidence_path('reports/expanded_v6/source_holdout/results.json').read_text(encoding='utf-8'))
    heldout=hold['kaz_roberta_last']['test']['macro_f1']
    counts=frame.groupby('split').size().to_dict()
    default=profile['models'][profile['default_model']]['name']
    content=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KazStyle — результаты и диссертация</title><style>{css}</style></head><body><main>
<h1>KazStyle: результаты и дальнейшая работа</h1><p class="muted">Эксперименты 6 октября 2026 · сверка проекта, статьи и ИПМ 18 октября 2026 · одна точка входа в проект.</p>
<p><strong>Работает локальный классификатор пяти стилей.</strong> Вставить текст: <a href="http://127.0.0.1:8765">стабильный сайт v4</a>. Если сервер выключен, запустить <code>start_site.cmd</code>. Основная модель — {html.escape(default)}, корпус {len(active_frame)} текстов. Внешний ИИ API не используется.</p>
<p>Новое исследование v6: <strong>{len(frame):,} текстов</strong>, по {len(frame)//5} на стиль. Train {counts['train']}, validation {counts['validation']}, test {counts['test']}. URL и сведения об источнике хранятся отдельно от признаков.</p>
<p class="note">V6 пока не заменяет стабильный сайт: выбранная по validation модель ошибается на знакомом школьном заявлении, которое v4 распознаёт правильно. <a href="http://127.0.0.1:8766">Отдельный экспериментальный сайт v6</a> позволяет проверить новые веса. Высокая общая метрика не отменяет эту регрессию.</p>
<div class="note">Это рабочие результаты на предварительных метках. Содержательную проверку пользователь поручил ассистенту; она обозначается AI-assisted curation. Для завершения нужны замороженная итоговая выборка, оценка на новых источниках с явно указанным происхождением меток и английский текст. Независимого человеческого gold сейчас нет.</div>
<h2>Открыть нужное</h2><div class="cards">{cards}</div>
{corpus_note}
{project_audit_note}
{audit_note}
<h2>Завершённое сравнение v6</h2><p>Среднее ± стандартное отклонение при seed 42, 43 и 44. Один фиксированный корпус; это не три независимых тестовых набора. По validation выбрана Kaz-RoBERTa last; решение о готовности к замене стабильного сайта принимается отдельно.</p><table><tr><th>Вариант</th><th>Validation Macro-F1</th><th>Внутренний test Macro-F1</th></tr>{rows}</table>
<p>Отдельно обученная Kaz-RoBERTa на диагностике переноса между источниками: <strong>{heldout:.4f} Macro-F1</strong>. Там другой train и другой состав test; напрямую вычитать это число из основной таблицы нельзя.</p>
{plan_note}
<h2>Проверка текстов и происхождение меток</h2><p>В новом аудите ассистент просмотрел 25 текстов, по пять на стиль, включая семь ограниченных выдержек. <a href="../configs/reviews/corpus_pilot_20261018.json">Решения и причины</a> сохранены отдельно; это не оценка точности всего корпуса и не финальный test. Проверка обнаружила удаление слов при сборе сказок и юбилейную заметку под научной меткой. Исправления сверены с сохранёнными первоисточниками.</p><p>Старые формы для людей остаются диагностическими; заполненных человеческих ответов нет. Дальнейшая проверка по выбору пользователя — AI-assisted curation. Для итогового эксперимента нужно заранее зарезервировать новые источники и группы, явно описать происхождение и ограничения меток. Коэффициент согласия людей без их ответов не рассчитывается.</p>
<p>В продолжении работы полностью прочитаны ещё 888 разговорных кандидатов: 810 приняты по стилю, 78 исключены или неоднозначны. Сборка <code>unified_corpus_20261018b</code> содержит 2 053 разговорных документа; каждый класс превысил промежуточный ориентир 2 000. Просмотр касается коротких текстов 8–55 слов и не доказывает качество всех жанров. Метрики новых моделей на этом банке ещё не получены.</p>
<p>Память и принятые решения: <a href="../docs/PROJECT.md">PROJECT.md</a>. Команды и архитектура: <a href="../docs/WORKFLOW.md">WORKFLOW.md</a>. Предыдущие эксперименты сохраняются в архиве; старый отчёт не является текущим результатом.</p>
<details><summary>Ограничения и доказательства</summary><p>Связь источника и класса остаётся сильной; длины классов различаются. Автоматическая проверка не доказывает правильность стилевой метки. В одном train-тексте v6 зафиксирован остаток подписи к фото: исправлен очиститель для следующей версии, замороженные данные не переписаны.</p><p>Сайт: <code>{html.escape(profile['dataset'])}</code><br>Исследование: <code>expanded_v6_shared</code><br>Manifest SHA-256 исследования: <code>{config['manifest_sha256']}</code></p></details>
</main></body></html>'''
    (reports/'PROJECT_STATUS.html').write_text(content,encoding='utf-8')
    write_json(technical/'PROJECT_STATUS.evidence.json',{'created_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':config['manifest_sha256'],'active_manifest_sha256':active_config['manifest_sha256'],'summary_sha256':file_hash(summary/'results.json'),'deployment_sha256':file_hash(root/'configs/deployment.json'),'human_reviews_received':0})
    print(json.dumps({'hub':str(reports/'PROJECT_STATUS.html'),'documents':len(frame),'default':profile['default_model']}))


if __name__=='__main__':build()
