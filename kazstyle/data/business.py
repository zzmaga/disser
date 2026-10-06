"""Collect bounded Kazakh business-document candidates from explicit sections.

Only the Kazakh document body is extracted. Page labels are provisional; teaching
instructions and mixed templates still need semantic review before training.
"""
import argparse
import json
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.forum import has_contact_identifier
from kazstyle.data.quality import clean_text, inspect_text

BASE = 'https://resmihat.kz/'
SECTIONS = {3: 'application_form', 13: 'personnel_order', 20: 'contract',
            194: 'application_form', 109: 'service_memo', 33: 'explanatory_note',
            34: 'service_memo', 181: 'explanatory_note', 182: 'business_form',
            14: 'order_extract', 15: 'administrative_order',
            205: 'procurement_order', 196: 'procurement_contract', 135: 'business_letter'}


def listing_links(listing, current_url):
    # Subsequent pages are JSON containing an HTML fragment and an explicit next URL.
    raw = str(listing).strip()
    next_from_json = None
    if raw.startswith('{'):
        payload = json.loads(raw)
        listing = BeautifulSoup(payload['html'], 'html.parser')
        next_from_json = payload.get('next') or None
    elif isinstance(listing, (str, bytes)):
        listing = BeautifulSoup(listing, 'html.parser')
    links = list(dict.fromkeys(urljoin(current_url, a['href']) for a in listing.select('.doc-item .name a[href]')))
    next_button = listing.select_one('[next-page-url]')
    next_url = urljoin(current_url, next_button['next-page-url']) if next_button else next_from_json
    if next_url:
        next_url = urljoin(current_url, next_url)
    return links, next_url


def parse_document(content):
    soup = BeautifulSoup(content, 'html.parser')
    body = soup.select_one('#integr_content_kz')
    title = soup.select_one('h1')
    if body is None or title is None:
        raise ValueError('Kazakh document body or title absent')
    for node in body.select('script,style,nav,form,button'):
        node.decompose()
    for node in body.select('br'):
        node.replace_with('\n')
    # Preserve inline fragments of a word; separate actual blocks and table cells.
    for node in body.select('p,div,li,tr,td,th,h1,h2,h3,h4'):
        node.append('\n')
    text = body.get_text('', strip=False)
    return text, title.get_text(' ', strip=True)


def collect(out, per_section=20, limit=150, cache_from=None):
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= per_section <= 50 or not 1 <= limit <= 300:
        raise ValueError('Collection bounds outside allowed range')
    out.mkdir(parents=True); cache = out / 'pages'; cache.mkdir()
    session = requests.Session(); agent = 'KazStyleResearch/1.0'
    session.headers['User-Agent'] = agent
    response = session.get(BASE + 'robots.txt', timeout=25)
    if response.status_code not in {200, 404}:
        raise ValueError(f'Cannot establish robots policy: {response.status_code}')
    robot = RobotFileParser(); robot.parse(response.text.splitlines() if response.status_code == 200 else [])
    (out / 'robots.txt').write_text(response.text, encoding='utf-8')
    last = 0.; blocked = False

    def fetch(url):
        nonlocal last, blocked
        if blocked:
            raise ValueError('Collection stopped after access/rate-limit response')
        if urlsplit(url).netloc != 'resmihat.kz' or not robot.can_fetch(agent, url):
            raise ValueError('Unexpected host or robots exclusion')
        name = digest(url) + '.html'
        if cache_from and (cache_from / name).is_file():
            path = cache / name
            shutil.copy2(cache_from / name, path)
            return path.read_text(encoding='utf-8'), path
        time.sleep(max(0., 1. - (time.monotonic() - last))); last = time.monotonic()
        with session.get(url, timeout=(8, 25), stream=True) as result:
            if result.status_code in {401, 403, 429}:
                blocked = True
            result.raise_for_status()
            if urlsplit(result.url).netloc != 'resmihat.kz':
                raise ValueError('Unexpected redirect host')
            parts = []; size = 0
            for chunk in result.iter_content(65536):
                size += len(chunk)
                if size > 4 * 1024 * 1024:
                    raise ValueError('Page byte limit exceeded')
                parts.append(chunk)
            raw = b''.join(parts); path = cache / (digest(url) + '.html'); path.write_bytes(raw)
            return raw.decode('utf-8'), path

    candidates, quarantine, errors, seen, hashes = [], [], [], set(), set()
    write_json(out / 'protocol.json', {'source': BASE, 'sections': SECTIONS,
        'per_section': per_section, 'max_documents': limit, 'interval_seconds': 1.,
        'body_selector': '#integr_content_kz', 'labels': 'source_section_not_expert',
        'cache_from': str(cache_from) if cache_from else None,
        'code_sha256': file_hash(__file__), 'used_for_training': False})
    for section, genre in SECTIONS.items():
        if blocked or len(seen) >= limit:
            break
        try:
            listing_url = BASE + f'documents/category/{section}'
            links = []; visited = set()
            for _ in range(10):
                if not listing_url or listing_url in visited or len(links) >= per_section:
                    break
                visited.add(listing_url)
                listing, _ = fetch(listing_url)
                found, listing_url = listing_links(listing, listing_url)
                links = list(dict.fromkeys([*links, *found]))
            count = 0
            for link in links:
                match = re.fullmatch(r'/documents/(\d+)', urlsplit(link).path)
                if not match or match[1] in seen:
                    continue
                if blocked or count >= per_section or len(seen) >= limit:
                    break
                seen.add(match[1]); count += 1
                try:
                    page, path = fetch(link); raw, title = parse_document(str(page))
                    text, cleaning = clean_text(raw)
                    stats, reasons = inspect_text(text, min_words=15, max_words=50000)
                    text_hash = digest(text.casefold())
                    if text_hash in hashes:
                        reasons.append('exact_duplicate')
                    if has_contact_identifier(text):
                        reasons.append('possible_contact_or_identifier')
                    row = {'doc_id': digest('resmihat_document:' + match[1])[:24],
                           'text': text, 'title': title, 'style': 'official', 'genre': genre,
                           'source_url': link, 'source_domain': 'resmihat.kz',
                           'parent_id': 'resmihat_document:' + match[1], 'source_section': section,
                           'source_snapshot': str(path), 'source_sha256': file_hash(path),
                           'content_hash': text_hash, 'quality': stats, 'cleaning': cleaning,
                           'eligibility_reasons': reasons, 'review_status': 'unreviewed',
                           'label_origin': 'business_document_section_NOT_verified_style',
                           'usage_status': 'review_pool_only',
                           'license': 'Publicly accessible document examples; no blanket redistribution permission asserted',
                           'collected_at': datetime.now(timezone.utc).isoformat()}
                    (quarantine if reasons else candidates).append(row)
                    hashes.add(text_hash)
                except (requests.RequestException, ValueError) as exc:
                    errors.append({'url': link, 'error': str(exc)})
            write_jsonl(out / 'candidates.jsonl', candidates)
            write_jsonl(out / 'quarantine.jsonl', quarantine)
            print(f'business section={section}: {len(candidates)} candidates, {len(quarantine)} quarantined', flush=True)
        except (requests.RequestException, ValueError) as exc:
            errors.append({'section': section, 'error': str(exc)})
    write_json(out / 'errors.json', errors)
    summary = {'candidates': len(candidates), 'quarantined': len(quarantine), 'errors': len(errors),
               'by_genre': dict(Counter(r['genre'] for r in candidates)),
               'reasons': dict(Counter(v for r in quarantine for v in r['eligibility_reasons'])),
               'used_for_training': False, 'expert_verified': False,
               'limitations': ['Some pages contain instructions alongside templates.',
                   'Related forms may share a template family; group before splitting.',
                   'Public availability does not establish unrestricted redistribution rights.']}
    write_json(out / 'collection.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--per-section', type=int, default=20)
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--cache-from', type=Path, help='Reuse saved public pages without new requests')
    args = parser.parse_args()
    collect(args.out_dir, args.per_section, args.limit, args.cache_from)
