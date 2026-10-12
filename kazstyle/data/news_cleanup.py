"""Opt-in corpus revision; never changes deployed inference or frozen datasets.

Source-specific rules remove observed page furniture. Ambiguous content is flagged
for review, not deleted or relabelled. Use full candidates before excerpt selection.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from kazstyle.data.quality import clean_text, inspect_text

VERSION = 'source_furniture_review_v2'
MONTHS = 'Қаңтар|Ақпан|Наурыз|Сәуір|Мамыр|Маусым|Шілде|Тамыз|Қыркүйек|Қазан|Қараша|Желтоқсан'
PUBLICATION_DATE = re.compile(
    rf'^\s*(?:\d+\s+\d+\s+пікір\s+)?\d{{1,2}}\s+(?:{MONTHS}),?\s+20\d{{2}}\s+сағат\s+\d{{1,2}}:\d{{2}}\s+', re.I)
PLURAL_CREDIT = re.compile(r'\bСуреттер\s*:\s*[^.!?\n]{0,160}?\s+алынды[.!?]', re.I)
BREADCRUMB = re.compile(r'^\s*-\s*Мұқаба\s*-\s*Жаңалықтар\s*-\s*', re.I)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def revise_text(text, source_domain):
    """Return text and an explicit removal log, without using a style label."""
    original = text
    removed = []

    def remove(pattern, value, rule):
        def replacement(match):
            removed.append({'rule': rule, 'removed_text': match.group(0)})
            return ' '
        return pattern.sub(replacement, value)

    domain = (source_domain or '').lower().removeprefix('www.')
    if domain == 'abai.kz':
        text = remove(PUBLICATION_DATE, text, 'abai_publication_timestamp')
        text = remove(PLURAL_CREDIT, text, 'abai_plural_photo_credit')
        marker = re.search(r'\bЕң көп оқылған\s+1\s+', text, re.I)
        if marker and re.search(r'\bҮздік материалдар\b', text[marker.end():], re.I):
            removed.append({'rule': 'abai_popular_and_recommended', 'removed_text': text[marker.start():]})
            text = text[:marker.start()]
    if domain == 'baq.kz':
        text = remove(BREADCRUMB, text, 'baq_navigation_prefix')
        marker = re.search(r'\bЕң оқылған:\s*-\s+', text, re.I)
        if marker and len(re.findall(r'\s-\s', text[marker.start():])) >= 2:
            removed.append({'rule': 'baq_popular_links', 'removed_text': text[marker.start():]})
            text = text[:marker.start()]
    if domain == 'egemen.kz':
        # Observed empty leading captions after address removal. Strip the label,
        # not the following sentence. Known residual credits are exact prefixes.
        prefix = re.match(r'^\s*Сурет:\s*', text, re.I)
        if prefix:
            text = remove(re.compile(r'^\s*Сурет:\s*', re.I), text, 'egemen_leading_photo_label').lstrip()
            for credit in ['polese.by', 'ашық дереккөзден', 'коллажы', '.ua',
                           'almaty.tv', 'mofa.go.jp', 'Қазақстан халқы Ассамблеясы']:
                if text.startswith(credit+' '):
                    removed.append({'rule': 'egemen_observed_photo_attribution', 'removed_text': credit})
                    text = text[len(credit):].lstrip()
                    break
        text = remove(re.compile(r'\(Фото:\s*[^()]{0,120}\)', re.I), text, 'egemen_parenthesized_photo_credit')
        text = remove(re.compile(r'^Басты фото:\s*Мәтін ішіндегі суреттер:\s*ТСМ\s+', re.I),
                      text, 'egemen_combined_photo_credit')
    if domain == 'adebiportal.kz':
        text = remove(re.compile(r'\s+Көрнекі сурет:\s*.{0,160}$', re.I), text, 'adebiportal_trailing_illustration_credit')
        text = remove(re.compile(r'\s+(?:Көрнекі )?[Сс]урет (?:ашық (?:ғаламтор|интернет) (?:кеңістігінен|көздерінен|алаңынан)|интернеттен|сайтынан) алынды\.?\s*$', re.I),
                      text, 'adebiportal_trailing_image_source')
        text = remove(re.compile(r'^\s*\(?\s*(?:әңгіме|деректі әңгіме)\)\s*', re.I),
                      text, 'adebiportal_parenthesized_genre_heading')
    text, base_changes = clean_text(text)
    return text, {'version': VERSION, 'changed': original != text,
                  'source_rules': removed, 'base_changes': base_changes}


def review_flags(text):
    """Screening only: a match is not proof of noise or an incorrect label."""
    flags = []
    if re.search(r'\b(?:Фото|Сурет(?:тер)?)(?:нің|тің)?\s*:|\bScreenshot\b', text, re.I):
        flags.append('possible_photo_credit')
    if re.search(r'Ең көп оқылған|Үздік материалдар|Ең оқылған:', text, re.I):
        flags.append('possible_recommendations')
    if text.count('|') >= 5:
        flags.append('table_not_continuous_prose')
    if re.search(r'\b\d+\s+пікір\b', text, re.I):
        flags.append('possible_comment_counter')
    return flags


def build(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'Use a new revision directory: {out}')
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if (not isinstance(row.get('text'), str) or not row.get('doc_id')
                or row['doc_id'] in ids):
            raise ValueError('Candidates need text and unique doc_id')
        if str(row.get('usage_status', '')).startswith('reserved'):
            raise ValueError('Reserved evaluation material cannot enter this revision')
        ids.add(row['doc_id'])
    revised, changes, flagged = [], [], []
    for row in rows:
        text, log = revise_text(row['text'], row.get('source_domain'))
        stats, reasons = inspect_text(text, min_words=8)
        flags = review_flags(text) + ['quality:'+r for r in reasons]
        updated = {**row, 'text': text, 'content_hash': digest(text), 'quality': stats,
                   'eligibility_reasons': reasons, 'cleanup_review_flags': flags,
                   'cleanup_revision': VERSION, 'upstream_text_sha256': digest(row['text']),
                   'upstream_review_status': row.get('review_status'),
                   'review_status': 'pending_review_after_cleanup' if log['changed'] else row.get('review_status')}
        # Flags do not silently exclude records or turn a provisional label into gold.
        revised.append(updated)
        if log['changed']:
            changes.append({'doc_id': row['doc_id'], 'source_domain': row.get('source_domain'),
                            'before': row['text'], 'after': text, **log})
        if flags:
            flagged.append({'doc_id': row['doc_id'], 'source_domain': row.get('source_domain'),
                            'style': row.get('style'), 'flags': flags, 'text': text})
    summary = {'version': VERSION, 'input': str(source), 'input_sha256': hashlib.sha256(raw).hexdigest(),
               'rows': len(rows), 'changed': len(changes), 'review_required': len(flagged),
               'source_rules': dict(Counter(x['rule'] for r in changes for x in r['source_rules'])),
               'by_style': dict(Counter(r['style'] for r in revised)),
               'expert_verified': False, 'ready_for_training': False,
               'note': 'Candidate revision only; no split or new experiment. Recheck groups and labels before training.'}
    out.mkdir(parents=True)
    for filename, records in [('candidates.jsonl', revised), ('changes.jsonl', changes), ('review_queue.jsonl', flagged)]:
        (out/filename).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    summary['candidates_sha256'] = hashlib.sha256((out/'candidates.jsonl').read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(__file__, out/'code_snapshot.py')
    (out/'base_cleaner_snapshot.py').write_bytes(Path(__file__).with_name('quality.py').read_bytes())
    (out/'COMPLETE.json').write_text(json.dumps({'candidates_sha256': summary['candidates_sha256']}), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.out_dir), ensure_ascii=False))
