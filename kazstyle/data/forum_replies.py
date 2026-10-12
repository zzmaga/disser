"""Extract whole answer/comment posts from verified saved forum pages, offline."""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.forum import BASE, has_contact_identifier, own_post_element, post_timestamps
from kazstyle.data.quality import clean_text, inspect_text


def parse_replies(raw, source_url):
    soup = BeautifulSoup(raw, 'html.parser')
    for kind, prefix in [('answer', 'a'), ('comment', 'c')]:
        for post in soup.select(f'.qa-{prefix}-list-item'):
            ident = post.get('id', '')
            if not re.fullmatch(prefix+r'\d+', ident):
                continue
            body = own_post_element(post, f'.qa-{prefix}-item-content [itemprop="text"]')
            if body is None:
                continue
            author = own_post_element(post, f'.qa-{prefix}-item-who-data .qa-user-link[href]')
            created, modified = post_timestamps(post)
            contains_quote_or_code = bool(body.select('blockquote,pre,code'))
            for node in body.select('script,style,nav,form,button'):
                node.decompose()
            yield {'post_id': ident, 'post_type': kind,
                   'text': body.get_text(' ', strip=True),
                   'source_url': source_url.split('#')[0]+'#'+ident,
                   'author_group': digest('surak_author:'+urljoin(BASE, author['href']))[:24] if author else None,
                   'published_at': created, 'modified_at': modified,
                   'contains_quote_or_code': contains_quote_or_code}


def build(sources, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = [{'path':str(p), 'sha256':file_hash(p)} for p in sources]
    accepted, rejected, seen_posts, seen_pages, seen_hashes = [], [], set(), set(), set()
    for source in sources:
        with Path(source).open(encoding='utf-8') as stream:
            for line in stream:
                question = json.loads(line)
                if str(question.get('usage_status', '')).startswith('reserved'):
                    raise ValueError('Reserved question threads cannot enter a development reply pool')
                url = question.get('source_url', '')
                match = re.search(r'/(\d+)/?', urlsplit(url).path)
                if urlsplit(url).netloc != 'surak.baribar.kz' or not match:
                    continue
                snapshot = Path(question['source_snapshot'])
                snapshot_sha = file_hash(snapshot)
                if snapshot_sha != question['source_sha256']:
                    raise ValueError(f'Source snapshot changed: {snapshot}')
                page_key = (match[1], snapshot_sha)
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
                for post in parse_replies(snapshot.read_bytes(), url):
                    key = post['post_id']
                    if key in seen_posts:
                        continue
                    seen_posts.add(key)
                    text, cleaning = clean_text(post.pop('text'))
                    stats, reasons = inspect_text(text, min_words=8, max_words=500)
                    sha = digest(' '.join(text.casefold().split()))
                    if has_contact_identifier(text):
                        reasons.append('possible_contact_or_identifier')
                    if post['contains_quote_or_code']:
                        reasons.append('quoted_or_code_block_requires_review')
                    published = post.get('published_at')
                    if published and published[:10] > date.today().isoformat():
                        reasons.append('future_publication_date')
                    if sha in seen_hashes:
                        reasons.append('normalized_exact_duplicate')
                    row = {**post, 'doc_id':digest('surak_'+key)[:24], 'text':text,
                           'style':'colloquial', 'genre':'forum_'+post['post_type'],
                           'parent_id':'surak_thread:'+match[1], 'source_domain':'surak.baribar.kz',
                           'source_snapshot':str(snapshot), 'source_sha256':snapshot_sha,
                           'content_hash':sha, 'quality':stats, 'cleaning':cleaning,
                           'eligibility_reasons':reasons, 'review_status':'unreviewed',
                           'label_origin':'forum_reply_genre_NOT_verified_style',
                           'usage_status':'review_pool_only',
                           'license':question.get('license', 'Original publisher rights apply')}
                    (rejected if reasons else accepted).append(row)
                    if not reasons:
                        seen_hashes.add(sha)
                if len(seen_pages) % 100 == 0:
                    print(f'Reply snapshots: {len(seen_pages)}; candidates: {len(accepted)}', flush=True)
    for item in manifest:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Input changed during offline reply extraction')
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl', accepted)
    write_jsonl(out/'quarantine.jsonl', rejected)
    report = {'inputs':manifest, 'pages':len(seen_pages), 'candidate_documents':len(accepted),
              'by_genre':dict(Counter(r['genre'] for r in accepted)), 'quarantined':len(rejected),
              'reasons':dict(Counter(reason for r in rejected for reason in r['eligibility_reasons'])),
              'expert_verified':False, 'used_for_training':False, 'network_requests':0,
              'unit':'one entire published user answer/comment; not a segment of a question',
              'limitations':['Semantic review required: replies may be instructions, quotations, essays or spam.',
                             'All question/answer/comment posts of one thread share the parent group.',
                             'Only replies present in the saved public page are available.']}
    write_json(out/'collection.json', report)
    shutil.copy2(__file__, out/'code_snapshot.py')
    write_json(out/'COMPLETE.json', {p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.sources, args.out_dir)
