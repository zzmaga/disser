"""Portable text-cleaning demo. No model weights, style labels or training."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from kazstyle.data.quality import CLEANING_VERSION, assert_text_only, clean_text
from kazstyle.settings import PROJECT_ROOT


def prepare(records):
    kept, excluded, seen_ids, seen_texts = [], [], set(), {}
    for number, row in enumerate(records, 1):
        if (not isinstance(row, dict) or set(row) != {'id', 'text'}
                or not isinstance(row['id'], str) or not row['id'].strip()
                or not isinstance(row['text'], str)):
            raise ValueError(f'Row {number}: expected nonempty string id and string text')
        if row['id'] in seen_ids:
            raise ValueError(f'Row {number}: duplicate id')
        seen_ids.add(row['id'])
        text, _ = clean_text(row['text'])
        if not text or not any(c.isalpha() for c in text):
            excluded.append({'id': row['id'], 'reason': 'empty_or_no_letters'})
            continue
        assert_text_only([text])
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest in seen_texts:
            excluded.append({'id': row['id'], 'reason': 'exact_duplicate',
                             'duplicate_of': seen_texts[digest]})
            continue
        seen_texts[digest] = row['id']
        kept.append({'id': row['id'], 'text': text, 'words': len(text.split()),
                     'text_sha256': digest})
    summary = {'purpose': 'Synthetic cleaning demonstration; not a style benchmark',
               'cleaning_version': CLEANING_VERSION, 'input_rows': len(seen_ids),
               'kept_rows': len(kept), 'excluded_rows': len(excluded),
               'excluded': excluded}
    return kept, summary


def build(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    kept, summary = prepare(records)
    summary['input_sha256'] = hashlib.sha256(raw).hexdigest()
    out.mkdir(parents=True)
    (out/'cleaned.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in kept), encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    rows = ''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+str(r['words'])+
                   '</td><td>'+html.escape(r['text'])+'</td></tr>' for r in kept)
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KazStyle: пример очистки</title><style>body{font:17px/1.6 system-ui;max-width:960px;margin:40px auto;padding:20px;color:#203447}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:12px;text-align:left}code{background:#eef3f7}</style>
<h1>KazStyle: очистка текста</h1><p>Это искусственные примеры для проверки кода. Здесь нет обучения или определения стиля.</p>'''
    page += f'<p>Входных строк: <b>{summary["input_rows"]}</b>. Сохранено: <b>{len(kept)}</b>. Исключено: <b>{len(summary["excluded"])}</b>.</p>'
    page += '<p>Удалены ссылки и HTML; точные повторы после очистки исключены. Казахские буквы, имена и даты сохраняются.</p><table><tr><th>ID</th><th>Слов</th><th>Очищенный текст</th></tr>'+rows+'</table><p>Данные: <a href="cleaned.jsonl">cleaned.jsonl</a>. Причины исключения и контрольная сумма входа: <a href="summary.json">summary.json</a>.</p></html>'
    (out/'report.html').write_text(page, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT/'examples/cleaning_input.jsonl')
    parser.add_argument('--out-dir', type=Path, default=Path('build/demo'))
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.out_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
