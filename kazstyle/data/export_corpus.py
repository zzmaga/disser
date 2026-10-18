"""Export an uncapped, versioned document corpus with text-only CSVs and provenance.

This is a development corpus, not a train/test split or expert-labelled benchmark.
Frozen source recipes remain unchanged. Suspect records are separated, not erased.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import html
import json
from pathlib import Path
import re
import shutil
import unicodedata

from kazstyle.data.corpus import STYLES, digest, file_hash, write_json, write_jsonl
from kazstyle.data.forum import has_contact_identifier
from kazstyle.data.news_cleanup import revise_text, review_flags
from kazstyle.data.quality import assert_text_only, inspect_text

VERSION = 'document_csv_v5'
SNAPSHOT_MODULES = ('export_corpus.py', 'news_cleanup.py', 'news_html.py', 'quality.py',
                    'forum.py', 'corpus.py', 'grouping.py', 'overlap.py', 'prose.py', 'folktales.py')
DEFAULT_TARGETS = {style: 2000 for style in STYLES}
ARTICLE_TARGETS = dict(official=4200, scientific=1800, publicistic=6500,
                       literary=2100, colloquial=8000)
# A conservative review signal for less common TLDs left by the deployed cleaner.
# Do not delete matches: filenames and technical notation can look like domains.
DOMAIN_LIKE = re.compile(r'(?<![\w@])(?:[a-z0-9][a-z0-9_-]*\.)+[a-z]{2,24}\b|'
                         r'(?<!\w)[a-z0-9][a-z0-9_-]*\s+\.(?:kz|ru|com|org|net)\b|'
                         r'(?<!\w)[a-z0-9][a-z0-9_-]*\.\s+(?:kz|ru|com|org|net|by|tv|jp)\b|'
                         r'\bwww\.\s+\w|'
                         r'(?<![\w@])(?:[\w-]+\.)+(?:kz|ru|com|org|net|қаз|рф|кз|ру|ком|орг)\b', re.I)


def normalized_hash(text):
    return digest(' '.join(unicodedata.normalize('NFKC', text).casefold().split()))


def publisher_groups(row):
    """Conservative publisher-attributed families; never verified identities."""
    if row.get('source_domain') != 'adebiportal.kz' or not isinstance(row.get('title'), str):
        return {}
    parts = re.split(r'[.:]', row['title'], maxsplit=1)
    author, title = (parts if len(parts) == 2 else ('', ''))
    tokens = author.split()
    if (not title.strip() or not 2 <= len(tokens) <= 4
            or not all(t[0].isupper() and all(c.isalpha() or c == '-' for c in t) for t in tokens)):
        return {}
    author_key = ' '.join(author.casefold().split())
    title_key = ' '.join(re.sub(r'\((?:әңгіме|хикаят|новелла)\)', '', title.casefold()).split()).strip(' «»"')
    return {'author_group': row.get('author_group') or digest('published_byline|byline|'+author_key)[:24],
            'work_group': row.get('work_group') or digest('publisher_work|'+author_key+'|'+title_key)[:24],
            'publisher_group_evidence': {'field':'title', 'author':author, 'title':title.strip(),
                                         'status':'publisher_attribution_not_verified_identity'}}


def read_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding='utf-8'))
    if recipe.get('purpose') != 'provisional_document_corpus':
        raise ValueError('Expected provisional_document_corpus recipe')
    targets = recipe.get('collection_targets', DEFAULT_TARGETS)
    if (not isinstance(targets, dict) or set(targets) != set(STYLES)
            or any(type(n) is not int or n <= 0 for n in targets.values())):
        raise ValueError('Collection targets must contain a positive integer for every style')
    filters = recipe.get('require_review_filters', [])
    if any(not f or not set(f).issubset({'style', 'source_domain', 'genre'}) for f in filters):
        raise ValueError('Invalid source/genre review filter')
    rows, by_id, repeats = [], {}, []
    for item in recipe['inputs']:
        source = Path(item['path'])
        if file_hash(source) != item['sha256']:
            raise ValueError(f'Source checksum mismatch: {source}')
        with source.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if (not isinstance(row.get('doc_id'), str) or not row['doc_id'].strip()
                        or not isinstance(row.get('text'), str) or row.get('style') not in STYLES):
                    raise ValueError(f'Invalid document schema: {source}:{line_number}')
                if str(row.get('usage_status', '')).startswith('reserved'):
                    raise ValueError('Reserved material cannot enter a development corpus')
                origin = {'path': str(source), 'line': line_number, 'text_sha256': digest(row['text'])}
                needs_review = bool(item.get('require_review') or row.get('requires_explicit_style_review')) or any(
                    all(row.get(k) == value for k, value in f.items()) for f in filters)
                if row['doc_id'] in by_id:
                    first = by_id[row['doc_id']]
                    if (first['text'], first['style']) != (row['text'], row['style']):
                        raise ValueError(f'Conflicting document versions: {row["doc_id"]}')
                    first['input_occurrences'].append(origin)
                    first['requires_semantic_review'] |= needs_review
                    repeats.append({'doc_id': row['doc_id'], **origin})
                    continue
                row = {**row, 'input_occurrences': [origin],
                       'requires_semantic_review': needs_review}
                rows.append(row); by_id[row['doc_id']] = row
    exclusions = recipe.get('exclusions', [])
    if len({r['doc_id'] for r in exclusions}) != len(exclusions):
        raise ValueError('Duplicate exclusion IDs')
    for decision in exclusions:
        row = by_id.get(decision['doc_id'])
        if row is None or digest(row['text']) != decision.get('text_sha256') or not decision.get('reason'):
            raise ValueError('Exclusion requires an existing ID, matching text hash and reason')
        row['explicit_exclusion'] = decision['reason']
    reviewed_ids = set()
    for item in recipe.get('reviews', []):
        review_path = Path(item['path'])
        if file_hash(review_path) != item['sha256']:
            raise ValueError('Review checksum mismatch')
        review = json.loads(review_path.read_text(encoding='utf-8'))
        if review.get('reviewer_kind') != 'assistant':
            raise ValueError('Independent human annotation must use the adjudication workflow')
        for decision in review['decisions']:
            row = by_id.get(decision['doc_id'])
            if (row is None or decision['doc_id'] in reviewed_ids
                    or digest(row['text']) != decision.get('text_sha256')
                    or type(decision.get('keep')) is not bool or not decision.get('reason')
                    or decision.get('reviewed_extent') not in {'entire_candidate_text', 'title_beginning_end'}):
                raise ValueError('Invalid, repeated or stale review decision')
            reviewed_ids.add(decision['doc_id'])
            edited=row['text']
            for field in ['remove_prefix','remove_suffix']:
                value=decision.get(field)
                if value is not None:
                    if row.get('source_extraction_status')=='article_prose_html':
                        raise ValueError('Review removals must target the final HTML extraction, not its upstream text')
                    if not decision['keep'] or not isinstance(value,str) or not value or len(value)>300:
                        raise ValueError('Invalid bounded review removal')
                    if field=='remove_prefix':
                        if not edited.startswith(value):raise ValueError('Reviewed prefix changed')
                        edited=edited[len(value):].lstrip()
                    else:
                        if not edited.endswith(value):raise ValueError('Reviewed suffix changed')
                        edited=edited[:-len(value)].rstrip()
            if not edited.strip():raise ValueError('Review removal emptied text')
            row['assistant_review'] = {**decision, 'review_file_sha256': item['sha256']}
            row['requires_semantic_review'] = False
            if decision['keep']:
                row['review_status'] = ('assistant_reviewed_not_expert' if decision['reviewed_extent'] == 'entire_candidate_text'
                                        else 'assistant_screened_not_expert')
                row['label_origin'] = 'source_plus_assistant_semantic_review'
            else:
                row['explicit_exclusion'] = decision['reason']
    repaired_ids = set()
    for decision in recipe.get('source_repairs', []):
        row = by_id.get(decision.get('doc_id'))
        kind=decision.get('kind')
        prose=kind=='verified_prose_soft_hyphens'
        folktale=kind=='verified_folktale_vocabulary'
        if (row is None or decision.get('doc_id') in repaired_ids
                or not (prose or folktale)
                or digest(row['text']) != decision.get('text_sha256')
                or (prose and (row.get('source_sha256') != decision.get('source_sha256')
                    or type(decision.get('soft_breaks')) is not int or decision['soft_breaks'] <= 0))
                or (folktale and (row.get('source_domain')!='ertegiler.kz'
                    or not isinstance(decision.get('source_snapshot'),str)
                    or not re.fullmatch('[a-f0-9]{64}',decision.get('source_sha256',''))
                    or type(decision.get('inline_word_buttons')) is not int or decision['inline_word_buttons']<=0))
                or not re.fullmatch('[a-f0-9]{64}', decision.get('repaired_text_sha256', ''))):
            raise ValueError('Invalid, repeated or stale source repair')
        repaired_ids.add(decision['doc_id'])
        row['source_repair'] = decision
    return recipe, rows, repeats


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def verify_export(out):
    """Round-trip quoted CSV, enforce text-only inputs and exact metadata alignment."""
    csv.field_size_limit(100_000_000)
    metadata = [json.loads(line) for line in (out/'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    with (out/'corpus.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['text', 'label']:
            raise ValueError('Unexpected training CSV columns')
        rows = list(reader)
    if len(rows) != len(metadata):
        raise ValueError('CSV/metadata length mismatch')
    seen, doc_ids = set(), set()
    for i, (row, meta) in enumerate(zip(rows, metadata), 1):
        if (set(row) != {'text', 'label'} or row['label'] not in STYLES.values()
                or meta['corpus_row'] != i or digest(row['text']) != meta['text_sha256']
                or row['label'] != STYLES[meta['style']]):
            raise ValueError(f'CSV/metadata mismatch at record {i}')
        if not isinstance(meta.get('doc_id'), str) or not meta['doc_id'] or meta['doc_id'] in doc_ids:
            raise ValueError('Missing or repeated document ID in metadata')
        doc_ids.add(meta['doc_id'])
        sha = normalized_hash(row['text'])
        if sha in seen:
            raise ValueError('Duplicate normalized text in corpus CSV')
        seen.add(sha)
        assert_text_only([row['text']])
    for style, label in STYLES.items():
        with (out/'by_style'/f'{style}.csv').open(encoding='utf-8-sig', newline='') as stream:
            part = list(csv.DictReader(stream))
        if part != [r for r in rows if r['label'] == label]:
            raise ValueError(f'Per-style CSV mismatch: {style}')
    return {'rows_round_tripped': len(rows), 'unique_normalized_texts': len(seen),
            'metadata_alignment': True, 'addresses_in_model_text': 0, 'per_style_csvs_match': True}


def verify_integrity(out):
    """Verify the completed files without repeatedly parsing every document."""
    out = Path(out).resolve()
    manifest = json.loads((out/'COMPLETE.json').read_text(encoding='utf-8'))
    for name, sha in manifest.items():
        path = (out/name).resolve()
        if not path.is_relative_to(out) or file_hash(path) != sha:
            raise ValueError(f'Corpus integrity check failed: {name}')
    for required in ['corpus.csv', 'metadata.jsonl', 'summary.json']:
        if required not in manifest:
            raise ValueError('Incomplete corpus manifest')
    return manifest


def load_documents(out):
    """Read verified CSV/metadata; this does not approve labels or a training split."""
    out = Path(out).resolve()
    verify_integrity(out)
    verify_export(out)
    metadata = [json.loads(line) for line in (out/'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    with (out/'corpus.csv').open(encoding='utf-8-sig', newline='') as stream:
        return [{**meta, 'text': row['text']} for meta, row in zip(metadata, csv.DictReader(stream))]


def render_report(out, summary, rows):
    names = dict(official='Официально-деловой', scientific='Научный', publicistic='Публицистический',
                 literary='Художественный', colloquial='Разговорный')
    table = ''
    for style, name in names.items():
        lengths = sorted(r['quality']['words'] for r in rows if r['style'] == style)
        count = len(lengths)
        q = lambda fraction: lengths[int((count-1)*fraction)] if count else 0
        table += (f'<tr><td><a href="by_style/{style}.csv">{name}</a></td><td>{count:,}</td>'
                  f'<td>{summary["collection_targets"][style]:,}</td><td>{summary["shortfall_by_style"][style]:,}</td>'
                  f'<td>{q(.1)} / {q(.5)} / {q(.9)}</td></tr>')
    sources = Counter((r['style'], r.get('source_domain', 'unknown')) for r in rows)
    source_table = ''.join(f'<tr><td>{names[s]}</td><td>{html.escape(host)}</td><td>{n}</td></tr>'
                           for (s, host), n in sorted(sources.items()))
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>KazStyle: единый корпус</title>
<style>body{{font:16px/1.55 system-ui;background:#f3f6fa;color:#183048;margin:0}}main{{max-width:1120px;margin:30px auto;background:white;padding:32px;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}a{{color:#175ca8}}.note{{background:#fff1cc;padding:16px}}code{{background:#eef2f6;padding:2px 5px}}</style><main>
<h1>Единый корпус KazStyle</h1>
<p><strong>{len(rows):,} документов</strong> в основном CSV. Отдельно: {summary['review']} записей для разбора и {summary['excluded']} исключённых записей. Нового обучения на этой версии ещё не было.</p>
<p><a href="corpus.csv">Открыть весь корпус — corpus.csv</a> · <a href="metadata.jsonl">Источники и группы</a> · <a href="review.csv">На разбор</a> · <a href="excluded.csv">Исключено</a> · <a href="summary.json">Проверки и числа</a></p>
<p class="note">Это предварительный исследовательский корпус. Автоматическая очистка и отбор помощником не заменяют независимую проверку стиля. Случайно делить строки на train/test нельзя: родственные документы должны оставаться вместе.</p>
<h2>Сколько текстов действительно есть</h2>
<table><tr><th>Стиль / отдельный CSV</th><th>После отбора</th><th>Текущий ориентир</th><th>Недостаёт по стилю</th><th>Слов: 10% / медиана / 90%</th></tr>{table}</table>
<p>Ориентир задаётся рецептом сбора и не подтверждает качество разметки или прежние эксперименты. Полный банк сохраняется: классы не уравниваются копированием строк или удалением лишних новостей. Длинные произведения сохранены целиком; фрагменты не считаются самостоятельными документами.</p>
<h2>Формат и происхождение</h2>
<p>Все основные CSV имеют ровно два столбца: <code>text,label</code>. Кодировка UTF-8 с BOM, разделитель — запятая. Метки: Formal, Scientific, Publicist, Artistic, Colloquial. Ссылки, названия сайтов и сведения о проверке находятся в метаданных и не являются отдельными входными признаками классификатора.</p>
<p><code>metadata.jsonl</code> содержит <code>doc_id</code>, исходный адрес, происхождение метки, контрольные суммы и номер записи <code>corpus_row</code> (начиная с 1, без заголовка). <code>audit.jsonl</code> сохраняет решение по каждому документу; <code>changes.jsonl</code> — журнал новой очистки. Исходные файлы не перезаписаны.</p>
<p>Длина и стиль источника всё ещё могут подсказывать класс. Перед сравнением моделей нужны проверка меток, подбор сопоставимых текстовых фрагментов и разбиение с учётом групп. Старые диагностические тексты не становятся независимым финальным test.</p>
<h2>Что проверено программно</h2><p>CSV прочитаны обратно без потерь; все строки совпали с метаданными и отдельными файлами стилей. Проверены непустые тексты, допустимые метки, точные нормализованные повторы и отсутствие адресов по правилам очистителя. Это не гарантия отсутствия любого вида мусора или парафразов.</p>
<p>Группировка: {html.escape(json.dumps(summary['grouping'], ensure_ascii=False))}</p>
<h2>Источники основного CSV</h2><table><tr><th>Стиль</th><th>Источник</th><th>Документы</th></tr>{source_table}</table>
<p>Корпус хранится локально. Лицензия MIT проекта не распространяется автоматически на эти тексты. Условия каждого источника сохранены в метаданных.</p>
</main></html>'''
    (out/'corpus.html').write_text(page, encoding='utf-8')


def build(recipe_path, out):
    recipe_path, out = Path(recipe_path), Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new corpus version: {out}')
    recipe_sha256=file_hash(recipe_path)
    source_hashes={name:file_hash(Path(__file__).with_name(name)) for name in SNAPSHOT_MODULES}
    recipe, rows, repeats = read_recipe(recipe_path)
    if file_hash(recipe_path)!=recipe_sha256:
        raise ValueError('Recipe changed while reading corpus inputs')
    processed, text_groups, changes = [], defaultdict(list), []
    for i, row in enumerate(rows):
        raw_text = row['text']
        manual_removed=[]
        if row.get('source_repair'):
            if row['source_repair']['kind']=='verified_folktale_vocabulary':
                from kazstyle.data.folktales import repair_verified_source
            else:
                from kazstyle.data.prose import repair_verified_source
            raw_text, repair_log = repair_verified_source(row, row['source_repair'])
            manual_removed.append(repair_log)
        for field in ['remove_prefix','remove_suffix']:
            value=(row.get('assistant_review') or {}).get(field)
            if value:
                if not (raw_text.startswith(value) if field=='remove_prefix' else raw_text.endswith(value)):
                    raise ValueError('Review boundary changed after source repair')
                raw_text=(raw_text[len(value):].lstrip() if field=='remove_prefix'
                          else raw_text[:-len(value)].rstrip())
                manual_removed.append({'rule':'hash_bound_assistant_'+field,'removed_text':value})
        html_removed = []
        if row.get('source_extraction_status') == 'article_prose_html':
            from kazstyle.data.news_html import parse_article, VERSION as HTML_VERSION
            if row.get('source_domain') != 'egemen.kz' or row.get('snapshot_compression') != 'gzip':
                raise ValueError('Unexpected HTML snapshot source/format')
            snapshot = Path(row['source_snapshot'])
            if file_hash(snapshot) != row['source_sha256']:
                raise ValueError('HTML snapshot checksum mismatch')
            with gzip.open(snapshot, 'rb') as stream:
                raw = stream.read(4*1024*1024+1)
            if len(raw) > 4*1024*1024:
                raise ValueError('HTML snapshot byte budget exceeded')
            try:
                raw_text, publisher, html_removed, html_cleaning = parse_article(raw, row['source_url'])
                row = {**row, 'publisher_metadata': publisher, 'html_extraction_version': HTML_VERSION,
                       'html_extraction_removed': html_removed, 'html_text_cleaning': html_cleaning}
            except ValueError as exc:
                # A changed layout or a page containing only recommendations is
                # review material, never a fallback to a silently accepted old body.
                row = {**row, 'html_extraction_error': str(exc), 'html_extraction_version': HTML_VERSION}
        text, cleaning = revise_text(raw_text, row.get('source_domain'), genre=row.get('genre'))
        cleaning['source_rules'].extend(manual_removed)
        cleaning['changed'] = text != row['text']
        if html_removed:
            cleaning['html_removed'] = html_removed
        stats, quality_reasons = inspect_text(text, min_words=8)
        flags = review_flags(text, original_text=raw_text)
        title=row.get('title') or row.get('metadata',{}).get('title','')
        if row['style']=='scientific' and re.search(r'[-–—]\s*\d{2,3}\s*жаста\b',title,re.I):
            flags.append('commemorative_title_requires_review')
        if DOMAIN_LIKE.search(text):
            flags.append('residual_domain_like_token')
        if has_contact_identifier(text):
            flags.append('possible_contact_or_identifier')
        if row['requires_semantic_review']:
            flags.append('source_genre_requires_semantic_review')
        if row.get('source_extraction_status') == 'reextraction_failed_review_required':
            flags.append('source_reextraction_failed')
        if row.get('html_extraction_error'):
            flags.append('html_extraction_failed')
        if (row.get('publisher_metadata') or {}).get('contains_table'):
            flags.append('html_table_requires_review')
        if row.get('explicit_exclusion'):
            quality_reasons.append('explicit_review_exclusion')
        updated = {**row, **publisher_groups(row), 'text': text, 'text_sha256': digest(text),
                   'normalized_text_sha256': normalized_hash(text), 'quality': stats,
                   'cleanup_version': VERSION, 'quality_reasons': quality_reasons,
                   'review_flags': flags, 'upstream_review_status': row.get('review_status'),
                   'review_status': 'pending_review_after_cleanup' if cleaning['changed'] else row.get('review_status', 'unreviewed')}
        if cleaning['changed']:
            changes.append({'doc_id': row['doc_id'], 'before_sha256': digest(row['text']),
                            'after_sha256': updated['text_sha256'], **cleaning})
        processed.append(updated)
        text_groups[updated['normalized_text_sha256']].append(updated)
        if (i+1) % 2000 == 0:
            print(f'Cleaned {i+1}/{len(rows)} documents', flush=True)
    for group in text_groups.values():
        if len({r['style'] for r in group}) > 1:
            for row in group:
                row['quality_reasons'].append('conflicting_style_for_same_text')
        else:
            # Prefer a clean occurrence over one already excluded/requiring review.
            group.sort(key=lambda r: (bool(r['quality_reasons']), bool(r['review_flags']), r['doc_id']))
            for row in group[1:]:
                row['duplicate_of'] = group[0]['doc_id']
                row['quality_reasons'].append('normalized_exact_duplicate')
            group[0]['duplicate_occurrences'] = [
                {k: v for k, v in r.items() if k not in {'text', 'duplicate_occurrences'}} for r in group[1:]]
    accepted, review, excluded = [], [], []
    for row in sorted(processed, key=lambda r: (list(STYLES).index(r['style']), r['doc_id'])):
        row['label'] = STYLES[row['style']]
        row['disposition'] = 'excluded' if row['quality_reasons'] else 'review' if row['review_flags'] else 'accepted_provisional'
        row['reasons'] = json.dumps(row['quality_reasons'] + row['review_flags'], ensure_ascii=False)
        (excluded if row['quality_reasons'] else review if row['review_flags'] else accepted).append(row)
    assert_text_only([r['text'] for r in accepted])
    # Group shared passages and known families on full documents BEFORE any split.
    # Shared passages may be quotations/templates; they are not automatic deletions.
    grouping = {'performed': False, 'note': 'Full passage audit required before a split'}
    edges = []
    if recipe.get('passage_grouping'):
        import pandas as pd
        from kazstyle.data.grouping import assign_passage_groups, assign_split_groups
        if accepted:
            print(f'Grouping passages in {len(accepted)} full documents', flush=True)
            frame = pd.DataFrame(accepted)
            frame['parent_group'] = [r.get('parent_id') or r['doc_id'] for r in accepted]
            frame['passage_group_id'], edges = assign_passage_groups(frame)
            frame['split_group_id'] = assign_split_groups(frame)
            for row, (_, group) in zip(accepted, frame.iterrows()):
                for key in ['parent_group', 'passage_group_id', 'split_group_id']:
                    # Pandas represents an absent passage group as NaN. Persist
                    # JSON null instead: NaN is not valid interoperable JSON.
                    row[key] = None if pd.isna(group[key]) else group[key]
            sizes = frame.groupby('split_group_id').size()
            grouping = {'performed': True, 'shared_passage_edges': len(edges),
                        'split_groups': len(sizes), 'largest_group_documents': int(sizes.max()),
                        'mixed_style_groups': int((frame.groupby('split_group_id').style.nunique() > 1).sum()),
                        'note': 'Conservative groups, not a proof against all paraphrases or pretraining overlap. Reaudit exact model excerpts before splitting.'}
    # Recheck immutable inputs before committing any completed corpus version.
    for item in [*recipe['inputs'], *recipe.get('reviews', [])]:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Source changed during export')
    if file_hash(recipe_path)!=recipe_sha256:
        raise ValueError('Recipe changed during export')
    if any(file_hash(Path(__file__).with_name(name))!=sha for name,sha in source_hashes.items()):
        raise ValueError('Source code changed during export')
    out.mkdir(parents=True); (out/'by_style').mkdir(); (out/'code').mkdir()
    write_csv(out/'corpus.csv', accepted, ['text', 'label'])
    for style in STYLES:
        write_csv(out/'by_style'/f'{style}.csv', [r for r in accepted if r['style'] == style], ['text', 'label'])
    for name, part in [('review', review), ('excluded', excluded)]:
        write_csv(out/f'{name}.csv', part, ['doc_id', 'text', 'label', 'reasons'])
    metadata = []
    for i, row in enumerate(accepted, 1):
        meta = {k: v for k, v in row.items() if k not in {'text', 'reasons'}}
        meta['corpus_row'] = i
        meta['grouping_status'] = 'full_document_passages_and_known_families' if grouping['performed'] else 'preserved_parent_author_work; full_passage_audit_required_before_split'
        metadata.append(meta)
    write_jsonl(out/'metadata.jsonl', metadata)
    write_jsonl(out/'audit.jsonl', [{k: v for k, v in r.items() if k not in {'text', 'reasons'}} for r in processed])
    write_jsonl(out/'changes.jsonl', changes)
    write_jsonl(out/'passage_edges.jsonl', edges)
    write_json(out/'repeated_inputs.json', repeats)
    shutil.copy2(recipe_path, out/'recipe.json')
    if file_hash(out/'recipe.json')!=recipe_sha256:
        raise ValueError('Recipe changed while saving its snapshot')
    for i, item in enumerate(recipe.get('reviews', [])):
        shutil.copy2(item['path'], out/f'assistant_review_{i+1}.json')
    for name in SNAPSHOT_MODULES:
        shutil.copy2(Path(__file__).with_name(name), out/'code'/name)
        if file_hash(out/'code'/name)!=source_hashes[name]:
            raise ValueError('Source code changed while saving its snapshot')
    counts = Counter(r['style'] for r in accepted)
    summary = {'version': VERSION, 'recipe_sha256':recipe_sha256,
               'source_code_sha256':source_hashes, 'documents_before_screening': len(rows),
               'accepted_provisional': len(accepted), 'by_style': dict(counts),
               'review': len(review), 'review_by_style': dict(Counter(r['style'] for r in review)),
               'excluded': len(excluded), 'excluded_reasons': dict(Counter(v for r in excluded for v in r['quality_reasons'])),
               'review_reasons': dict(Counter(v for r in review for v in r['review_flags'])),
               'changed_documents': len(changes), 'repeated_input_records': len(repeats),
               'article_targets_reference_only': ARTICLE_TARGETS,
               'collection_targets': recipe.get('collection_targets', DEFAULT_TARGETS),
               'shortfall_by_style': {s: max(0, recipe.get('collection_targets', DEFAULT_TARGETS)[s]-counts[s]) for s in STYLES},
               'max_per_style': None, 'expert_verified': False, 'ready_for_training': False,
               'grouping': grouping,
               'new_final_test': False, 'models_used_for_selection': False,
               'limitations': ['Provisional source/assistant labels, not independent expert annotation.',
                   'Full documents retain natural length; fragments are not counted as new documents.',
                   'Exact duplicates removed; groups must be respected and exact model excerpts audited before a split.',
                   'Previously inspected development texts cannot become a new independent final test.',
                   'Rights stay with original sources; CSVs are local, not automatically publishable.'],
               'validation': verify_export(out)}
    write_json(out/'summary.json', summary)
    render_report(out, summary, accepted)
    write_json(out/'COMPLETE.json', {str(p.relative_to(out)): file_hash(p)
                                    for p in sorted(out.rglob('*')) if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--recipe', type=Path)
    mode.add_argument('--verify-dir', type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if args.verify_dir:
        records = load_documents(args.verify_dir)
        print(json.dumps({'integrity_verified': True, 'documents': len(records)}))
    else:
        if args.out_dir is None:
            parser.error('--recipe requires --out-dir')
        print(json.dumps(build(args.recipe, args.out_dir), ensure_ascii=False))
