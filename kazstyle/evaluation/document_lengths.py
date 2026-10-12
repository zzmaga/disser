"""Read-only CSV structure, document length and deployed excerpt coverage audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import statistics


def csv_structure(path):
    """Count logical records independently of physical lines; never repair input."""
    path = Path(path)
    csv.field_size_limit(100_000_000)
    result = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'records': 0, 'multiline_records': 0, 'invalid_width': [], 'error': None}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        try:
            result['columns'] = next(reader)
            previous = reader.line_num
            for record in reader:
                result['records'] += 1
                if len(record) != len(result['columns']):
                    result['invalid_width'].append({'record': result['records'], 'fields': len(record)})
                if reader.line_num - previous > 1:
                    result['multiline_records'] += 1
                previous = reader.line_num
        except (csv.Error, UnicodeError, StopIteration) as exc:
            result['error'] = str(exc) or 'empty CSV'
        result['physical_lines'] = reader.line_num
    result['valid_structure'] = not result['error'] and not result['invalid_width']
    return result


def describe(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    return {'n': len(values), 'min': values[0], 'p10': values[int((len(values)-1)*.1)],
            'median': statistics.median(values), 'p90': values[int((len(values)-1)*.9)],
            'p99': values[int((len(values)-1)*.99)], 'max': values[-1],
            'mean': round(statistics.mean(values), 3), 'sum': sum(values)}


def length_record(row):
    text = row['text']
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    long_lines = [line for line in lines if len(line.split()) >= 10]
    repeated = sum(n-1 for n in Counter(long_lines).values() if n > 1)
    return {'doc_id': row['doc_id'], 'style': row['style'], 'genre': row.get('genre', 'unknown'),
            'source_url': row.get('source_url'),
            'title': row.get('title', row.get('metadata', {}).get('title', '')),
            'words': len(text.split()), 'characters': len(text), 'nonempty_lines': len(lines),
            'max_line_characters': max(map(len, lines), default=0),
            'repeated_long_lines': repeated,
            'replacement_characters': text.count('\ufffd'),
            'literal_backslash_n': text.count('\\n'),
            'html_tag_fragments': len(re.findall(r'</?(?:html|body|div|script|table)\b', text, re.I))}


def build(corpus, dataset, out, legacy_dir=None):
    from kazstyle.data.export_corpus import load_documents
    from kazstyle.data.corpus import make_excerpt, write_json, write_jsonl
    from kazstyle.models.tokenization import tokenizer_specs, load_dataset_tokenizer, fit_shared_text
    corpus, dataset, out = Path(corpus), Path(dataset), Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows = load_documents(corpus)
    config = json.loads((dataset/'config.json').read_text(encoding='utf-8'))
    tokenizers = [load_dataset_tokenizer(dataset, config, spec['model_name'])
                  for spec in tokenizer_specs(config)]
    constraints = [(tok, spec['max_tokens']) for tok, spec in tokenizers]
    files = [corpus/'corpus.csv', *sorted((corpus/'by_style').glob('*.csv'))]
    if legacy_dir:
        files += sorted(Path(legacy_dir).glob('*.csv'))
    structures = [csv_structure(path) for path in files]
    measurements, by_style = [], defaultdict(list)
    for i, row in enumerate(rows, 1):
        item = length_record(row)
        excerpt, _ = make_excerpt(row['text'], tokenizers[0][0], config['max_words'], config['max_tokens'])
        excerpt = fit_shared_text(excerpt, constraints)
        item['excerpt_words'] = len(excerpt.split())
        item['word_coverage'] = item['excerpt_words'] / item['words']
        item['partial_text'] = excerpt != row['text']
        item['full_token_lengths'] = {
            spec['model_name']: len(tok(row['text'], add_special_tokens=True,
                                      truncation=False, verbose=False)['input_ids'])
            for tok, spec in tokenizers}
        measurements.append(item); by_style[row['style']].append(item)
        if i % 1000 == 0:
            print(f'Length audit: {i}/{len(rows)}', flush=True)
    summaries = {}
    for style, items in by_style.items():
        summaries[style] = {field: describe([r[field] for r in items])
                            for field in ('words', 'characters', 'nonempty_lines', 'excerpt_words', 'word_coverage')}
        summaries[style].update({'documents': len(items), 'partial_documents': sum(r['partial_text'] for r in items),
            'over_1000_words': sum(r['words'] > 1000 for r in items),
            'over_5000_words': sum(r['words'] > 5000 for r in items),
            'over_excel_cell_limit': sum(r['characters'] > 32767 for r in items),
            'full_token_lengths': {spec['model_name']: describe([r['full_token_lengths'][spec['model_name']] for r in items])
                                  for _, spec in tokenizers}})
    flags = [r for r in measurements if r['words'] > 5000 or r['repeated_long_lines'] >= 3
             or r['replacement_characters'] or r['literal_backslash_n'] or r['html_tag_fragments']]
    report = {'corpus': str(corpus), 'dataset_excerpt_policy': str(dataset), 'documents': len(rows),
              'csv_structure': structures, 'by_style': summaries, 'flagged_for_content_review': len(flags),
              'limitations': ['Length and repeated lines are review signals, not automatic rejection.',
                              'Full-document token lengths differ by tokenizer.',
                              'Coverage uses the deployed excerpt procedure on stored corpus text; no model inference.',
                              'A physical newline inside a quoted CSV field does not start another document.']}
    genres = defaultdict(list)
    for item in measurements:
        genres[(item['style'], item['genre'])].append(item)
    report['by_style_and_genre'] = [
        {'style':style, 'genre':genre, 'documents':len(items),
         'words':describe([r['words'] for r in items]),
         'word_coverage':describe([r['word_coverage'] for r in items])}
        for (style, genre), items in sorted(genres.items())]
    out.mkdir(parents=True)
    write_json(out/'summary.json', report)
    write_jsonl(out/'documents.jsonl', measurements)
    write_jsonl(out/'review_queue.jsonl', flags)
    names = {'official':'Официально-деловой', 'scientific':'Научный', 'publicistic':'Публицистический',
             'literary':'Художественный', 'colloquial':'Разговорный'}
    table = ''.join(f'<tr><td>{names[s]}</td><td>{v["documents"]}</td><td>{v["words"]["median"]:g}</td>'
                    f'<td>{v["words"]["p90"]}</td><td>{v["words"]["max"]}</td>'
                    f'<td>{v["excerpt_words"]["median"]:g}</td><td>{100*v["word_coverage"]["median"]:.1f}%</td></tr>'
                    for s,v in summaries.items())
    structure = ''.join(f'<tr><td>{html.escape(v["path"])}</td><td>{v["records"]}</td>'
                        f'<td>{v["physical_lines"]}</td><td>{v["multiline_records"]}</td><td>{v["valid_structure"]}</td></tr>'
                        for v in structures)
    samples = []
    row_map = {r['doc_id']: r for r in rows}
    for style in by_style:
        for item in sorted(by_style[style], key=lambda r:r['words'], reverse=True)[:3]:
            row = row_map[item['doc_id']]
            samples.append('<details><summary>'+html.escape(f'{names[style]}: {item["words"]} слов — {item["title"]}')
                           +'</summary><p>Начало и конец документа; это не полная содержательная проверка.</p><pre>'
                           +html.escape(row['text'][:1800]+'\n[…]\n'+row['text'][-1200:])+'</pre></details>')
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Длина документов и CSV</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap}details{margin:14px 0}</style>'
    page += '<h1>Длина документов и структура CSV</h1><p>Длинный документ сохраняется целиком в корпусе. Нынешний классификатор получает один общий центральный фрагмент, ограниченный словами и двумя токенизаторами. Число строк файла не равно числу документов.</p>'
    page += '<table><tr><th>Стиль</th><th>Документов</th><th>Медиана слов</th><th>90-й процентиль</th><th>Максимум</th><th>Медиана слов во входе</th><th>Медиана доли прочитанных слов</th></tr>'+table+'</table>'
    genre_table = ''.join(f'<tr><td>{names[r["style"]]}</td><td>{html.escape(str(r["genre"]))}</td>'
                          f'<td>{r["documents"]}</td><td>{r["words"]["median"]:g}</td>'
                          f'<td>{100*r["word_coverage"]["median"]:.1f}%</td></tr>'
                          for r in report['by_style_and_genre'])
    page += '<h2>Длины внутри жанров</h2><p>Добавление коротких стихотворений меняет общую медиану художественного класса. Это не означает, что длинная проза стала короче.</p><table><tr><th>Стиль</th><th>Жанр</th><th>Документов</th><th>Медиана слов</th><th>Медиана доли прочитанных слов</th></tr>'+genre_table+'</table>'
    page += '<h2>Записи и физические строки</h2><table><tr><th>Файл</th><th>Записей</th><th>Строк</th><th>Многострочных записей</th><th>Структура корректна</th></tr>'+structure+'</table>'
    page += '<p>Проверка структуры не подтверждает правильность меток. В исходных старых CSV могут быть другие столбцы; модельные входы берутся из завершённого корпуса. Тексты свыше 32 767 символов могут быть неудобны для ячейки Excel, но допустимы в CSV/Python.</p><h2>Самые длинные документы каждого стиля</h2>'+''.join(samples)
    (out/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({'documents':len(rows), 'review_flags':len(flags), 'report':str(out/'report.html')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--legacy-dir', type=Path)
    args = parser.parse_args()
    build(args.corpus, args.dataset, args.out_dir, args.legacy_dir)
