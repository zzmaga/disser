"""Re-extract Egemen article bodies from public HTML, retaining immutable evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import threading
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.news_html import parse_article, VERSION as HTML_VERSION


def refresh(source, out, limit=3000, workers=3):
    source, out = Path(source), Path(out)
    if (out/'COMPLETE.json').exists():
        raise FileExistsError(out)
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    source_sha = file_hash(source)
    selected = [r for r in rows if r.get('source_domain', '').removeprefix('www.') == 'egemen.kz']
    if not 1 <= limit <= 5000 or not 1 <= workers <= 3:
        raise ValueError('Invalid fetch limit')
    if out.exists() and not (out/'updates.jsonl').exists():
        raise ValueError('Existing directory is not a resumable collection')
    out.mkdir(parents=True, exist_ok=True); (out/'pages').mkdir(exist_ok=True)
    s = requests.Session(); agent = 'KazStyleResearch/1.0 (academic corpus cleanup)'
    s.headers['User-Agent'] = agent
    robot = RobotFileParser()
    r = s.get('https://egemen.kz/robots.txt', timeout=(8, 20))
    if r.status_code in {401, 403, 429}:
        raise ValueError('Robots access refused; collection stopped')
    if r.status_code >= 500:
        raise ValueError('Robots temporarily unavailable')
    robot.parse(r.text.splitlines() if r.status_code == 200 else [])
    (out/'robots.txt').write_text(r.text, encoding='utf-8')
    replacements, errors, rules = {}, [], Counter()
    originals = {r['doc_id']: r for r in selected}
    if (out/'updates.jsonl').exists():
        for line in (out/'updates.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if (row['doc_id'] not in originals or row['doc_id'] in replacements
                    or row['upstream_input_sha256'] != source_sha
                    or row['upstream_text_sha256'] != digest(originals[row['doc_id']]['text'])
                    or file_hash(row['source_snapshot']) != row['source_sha256']):
                raise ValueError('Invalid checkpoint provenance')
            replacements[row['doc_id']] = row
            rules.update(v['kind'] for v in row['removed_page_elements'])
    write_json(out/'protocol.json', {'source': str(source), 'source_sha256': source_sha,
               'workers': workers, 'minimum_request_start_interval_seconds': .7, 'limit': limit})
    blocked = threading.Event(); rate_lock = threading.Lock(); last = [0.]
    local = threading.local()
    def fetch_one(row):
            try:
                if blocked.is_set():
                    raise ValueError('Not fetched after limit/access stop')
                url = row['source_url']
                if urlsplit(url).netloc not in {'egemen.kz', 'www.egemen.kz'} or not robot.can_fetch(agent, url):
                    raise ValueError('Unexpected or robots-disallowed URL')
                with rate_lock:
                    time.sleep(max(0., .7-(time.monotonic()-last[0]))); last[0] = time.monotonic()
                    if blocked.is_set():
                        raise ValueError('Collection stopped after access/rate-limit response')
                if not hasattr(local, 'session'):
                    local.session = requests.Session(); local.session.headers['User-Agent'] = agent
                with local.session.get(url, timeout=(8, 20), stream=True) as response:
                    if response.status_code in {401, 403, 429}:
                        blocked.set()
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 4*1024*1024:
                            raise ValueError('Page byte limit exceeded')
                        chunks.append(chunk)
                    raw = b''.join(chunks)
                text, metadata, removed, cleaning = parse_article(raw, url)
                snapshot = out/'pages'/(row['doc_id']+'.html.gz')
                snapshot.write_bytes(gzip.compress(raw, mtime=0))
                updated = {**row, 'text': text, 'upstream_text_sha256': digest(row['text']),
                           'upstream_input_file': str(source), 'upstream_input_sha256': source_sha,
                           'source_snapshot': str(snapshot), 'source_sha256': file_hash(snapshot),
                           'snapshot_compression': 'gzip', 'source_html_sha256': hashlib.sha256(raw).hexdigest(),
                           'retrieved_at': datetime.now(timezone.utc).isoformat(),
                           'publisher_metadata': metadata, 'removed_page_elements': removed,
                           'cleaning': cleaning, 'content_hash': digest(text.casefold()),
                           'review_status': 'unreviewed_after_html_reextraction',
                           'source_extraction_status': 'article_prose_html',
                           'license': 'Publisher rights apply; original aggregate attribution retained in upstream metadata',
                           'upstream_license': row.get('license')}
                return updated, None
            except (requests.RequestException, ValueError) as exc:
                return None, {'doc_id': row['doc_id'], 'url': row['source_url'], 'error': str(exc)}
    pending = [r for r in selected[:limit] if r['doc_id'] not in replacements]
    print(f'News HTML: resuming {len(replacements)}; pending {len(pending)}', flush=True)
    with (out/'updates.jsonl').open('a', encoding='utf-8') as checkpoint, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, row) for row in pending]
        for i, future in enumerate(as_completed(futures), 1):
            updated, error = future.result()
            if error:
                errors.append(error)
            else:
                replacements[updated['doc_id']] = updated
                rules.update(item['kind'] for item in updated['removed_page_elements'])
                checkpoint.write(json.dumps(updated, ensure_ascii=False)+'\n'); checkpoint.flush()
            if i % 100 == 0:
                print(f'News HTML: extracted={len(replacements)}/{len(selected)}, errors={len(errors)}', flush=True)
    errors.extend({'doc_id': r['doc_id'], 'url': r['source_url'], 'error': 'Collection document limit'} for r in selected[limit:] if r['doc_id'] not in replacements)
    revised = []
    for row in rows:
        if row['doc_id'] in replacements:
            revised.append(replacements[row['doc_id']])
        elif row.get('source_domain', '').removeprefix('www.') == 'egemen.kz':
            revised.append({**row, 'source_extraction_status': 'reextraction_failed_review_required'})
        else:
            revised.append(row)
    if file_hash(source) != source_sha:
        raise ValueError('Source changed during refresh')
    write_jsonl(out/'candidates.jsonl', revised)
    summary = {'source': str(source), 'source_sha256': source_sha, 'eligible_for_refresh': len(selected),
               'reextracted': len(replacements), 'errors': errors, 'removed_elements': dict(rules),
               'total_preserved_records': len(revised), 'expert_verified': False,
               'extraction_version': HTML_VERSION}
    write_json(out/'collection.json', summary); shutil.copy2(__file__, out/'code_snapshot.py')
    for name in ['news_html.py', 'news_cleanup.py', 'quality.py']:
        shutil.copy2(Path(__file__).with_name(name), out/('snapshot_'+name))
    write_json(out/'COMPLETE.json', {'candidates_sha256': file_hash(out/'candidates.jsonl')})
    print(json.dumps({k: v for k, v in summary.items() if k != 'errors'}), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=3000)
    a = p.parse_args(); refresh(a.input, a.out_dir, a.limit)
