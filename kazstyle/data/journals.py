"""Bounded collection of Kazakh OJS abstracts into an unreviewed candidate pool."""
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone, date
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.quality import clean_text, inspect_text

ARCHIVES = {
    'pedagogy': 'https://bulletin-pedagogic-sc.kaznu.kz/index.php/1-ped/issue/archive',
    'mathematics_computing': 'https://bm.kaznu.kz/index.php/kaznu/issue/archive',
    'philosophy_politics': 'https://bulletin-philospolit.kaznu.kz/index.php/1-pol/issue/archive',
}


def article_key(url):
    match = re.search(r'/article/view/(\d+)', url)
    return urlsplit(url).netloc.lower() + '/article/' + match[1] if match else url


def parse_abstract(content):
    soup = BeautifulSoup(content, 'html.parser')
    abstract = soup.select_one('.item.abstract')
    if abstract is None:
        raise ValueError('No abstract section')
    for node in abstract.select('h1,h2,h3,script,style,.label'):
        node.decompose()
    text = abstract.get_text(' ', strip=True)
    text = re.split(r'(?:Түйін\s*сөздер|Кілт\s*сөздер|Ключевые\s*слова|Keywords)\s*:', text, maxsplit=1, flags=re.I)[0]
    def meta(name):
        return [m.get('content', '') for m in soup.select('meta[name]') if m['name'] == name]
    licenses = sorted({urljoin('https://creativecommons.org', a['href']) for a in soup.select('a[href]')
                       if re.match(r'https?://creativecommons.org/licenses/', a['href'])})
    return text, {'title': next(iter(meta('citation_title')), ''), 'authors': meta('citation_author'),
                  'doi': next(iter(meta('citation_doi')), ''),
                  'publication_date': next(iter(meta('citation_date')), ''),
                  'license_urls': licenses, 'page_language': soup.html.get('lang', '') if soup.html else ''}


def collect(out, per_journal=40, max_issues=8, excluded_samples=None):
    if not 1 <= per_journal <= 200 or not 1 <= max_issues <= 20:
        raise ValueError('Collection limits out of range')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cache = out / 'pages'; cache.mkdir()
    excluded = set()
    if excluded_samples:
        excluded = {article_key(r['source_url']) for r in json.loads(excluded_samples.read_text(encoding='utf-8'))}
    session = requests.Session(); session.headers['User-Agent'] = 'KazStyleResearch/1.0 (bounded academic corpus collection)'
    blocked, errors, accepted, rejected, seen = set(), [], [], [], set()
    last_request = 0.
    def fetch(url):
        nonlocal last_request
        host = urlsplit(url).netloc
        if host in blocked:
            raise ValueError('Host stopped after access/rate-limit response')
        time.sleep(max(0., .7 - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        with session.get(url, timeout=(8, 22), stream=True) as response:
            if response.status_code in {401,403,429}:
                blocked.add(host)
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 4 * 1024 * 1024:
                    raise ValueError('Page exceeds collection byte limit')
                chunks.append(chunk)
            content = b''.join(chunks)
            path = cache / (digest(url)+'.html'); path.write_bytes(content)
            return BeautifulSoup(content, 'html.parser'), response.url, path
    for journal, archive in ARCHIVES.items():
        host, count = urlsplit(archive).netloc, 0
        try:
            listing, base, _ = fetch(archive)
            issues = list(dict.fromkeys(urljoin(base,a['href']) for a in listing.select('a[href]') if '/issue/view/' in a['href']))[:max_issues]
            for issue in issues:
                if count >= per_journal or host in blocked: break
                listing, base, _ = fetch(issue)
                links = list(dict.fromkeys(urljoin(base,a['href']) for a in listing.select('a[href]') if re.search(r'/article/view/\d+/?$', a['href'])))
                for url in links:
                    if count >= per_journal or host in blocked: break
                    key = article_key(url)
                    if key in seen or key in excluded: continue
                    seen.add(key)
                    try:
                        page, final_url, path = fetch(url)
                        if not str(page.html.get('lang', '') if page.html else '').startswith('kk'):
                            locale = next((a['href'] for a in page.select('a[href]') if '/user/setLocale/kk' in a['href']), None)
                            if not locale: raise ValueError('No public Kazakh locale link')
                            page, final_url, path = fetch(urljoin(final_url,locale))
                        raw, metadata = parse_abstract(str(page))
                        text, cleaning = clean_text(raw)
                        stats, reasons = inspect_text(text, min_words=35, max_words=1500)
                        published = metadata['publication_date'].replace('/', '-')
                        if re.match(r'^\d{4}-\d{2}-\d{2}', published) and published[:10] > date.today().isoformat():
                            reasons.append('future_publication_date')
                        row = {'doc_id':digest(key)[:24], 'text':text, 'style':'scientific',
                               'parent_id':key, 'source_url':final_url, 'source_domain':host,
                               'journal':journal, 'genre':'research_abstract', 'metadata':metadata,
                               'label_origin':'journal_abstract_not_expert', 'review_status':'unreviewed',
                               'usage_status':'review_pool_only', 'license':'See metadata.license_urls; no blanket redistribution permission asserted',
                               'collected_at':datetime.now(timezone.utc).isoformat(),
                               'source_snapshot':str(path), 'source_sha256':file_hash(path),
                               'content_hash':digest(text.casefold()), 'cleaning':cleaning,
                               'quality':stats, 'eligibility_reasons':reasons}
                        (rejected if reasons else accepted).append(row)
                        if not reasons: count += 1
                    except (requests.RequestException, ValueError) as exc:
                        errors.append({'url':url, 'error':str(exc)})
                print(f'[journals] {journal}: accepted={count}, seen={len(seen)}',flush=True)
                # Incremental checkpoint, so interrupted collection leaves usable provenance.
                write_jsonl(out/'candidates.jsonl', accepted)
                write_jsonl(out/'quarantine.jsonl', rejected)
        except (requests.RequestException, ValueError) as exc:
            errors.append({'url':archive, 'error':str(exc)})
    write_jsonl(out/'candidates.jsonl', accepted)
    write_jsonl(out/'quarantine.jsonl', rejected)
    report = {'candidate_documents':len(accepted), 'by_journal':dict(Counter(r['journal'] for r in accepted)),
              'quarantined':len(rejected), 'quarantine_reasons':dict(Counter(v for r in rejected for v in r['eligibility_reasons'])),
              'excluded_probe_articles':sorted(excluded), 'blocked_hosts':sorted(blocked), 'errors':errors,
              'expert_verified':False, 'used_for_training':False}
    write_json(out/'collection.json',report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--per-journal',type=int,default=40)
    parser.add_argument('--max-issues',type=int,default=8)
    parser.add_argument('--exclude-probe',type=Path)
    args=parser.parse_args()
    collect(args.out_dir,args.per_journal,args.max_issues,args.exclude_probe)
