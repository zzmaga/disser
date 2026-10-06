"""Blind corpus-audit packets and strict comparison of two independent reviews.

Packets audit candidate labels, not generalization to unseen sources. No model
is queried; agreement never automatically promotes documents into training.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from kazstyle.data.corpus import STYLES, digest, file_hash, write_json, write_jsonl

LABELS = {**{k:v for k,v in zip(STYLES,['Официально-деловой','Научный','Публицистический','Художественный','Разговорный'])},
          'mixed':'Смешанный: нет преобладающего стиля', 'insufficient':'Недостаточно контекста',
          'non_kazakh':'Не казахский текст', 'noise':'Мусор / повреждённый текст'}


def packet_hash(items):
    return digest(json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def read_packet(path):
    packet = json.loads(path.read_text(encoding='utf-8'))
    items = packet['items']
    if not items or len({r['id'] for r in items}) != len(items):
        raise ValueError('Empty packet or duplicate item IDs')
    if packet_hash(items) != packet['packet_sha256']:
        raise ValueError('Packet content hash mismatch')
    if any(digest(r['text']) != r['text_sha256'] for r in items):
        raise ValueError('Packet text hash mismatch')
    return packet


def review_excerpt(text, max_words=350):
    words = text.split()
    if len(words) <= max_words:
        return text, False
    # Annotators judge only the shown excerpt, not an unseen full document.
    start = (len(words)-max_words)//2
    return ' '.join(words[start:start+max_words]), True


def select_balanced(rows, per_style, seed):
    rng = random.Random(seed)
    selected = []
    for style in STYLES:
        buckets = defaultdict(list)
        for row in rows:
            if row['style'] != style: continue
            length = len(row['text'].split())
            band = 'short' if length < 40 else 'medium' if length < 150 else 'long'
            buckets[(row.get('source_domain','unknown'),row.get('genre','unknown'),band)].append(row)
        for values in buckets.values(): rng.shuffle(values)
        queues = [deque(values) for _,values in sorted(buckets.items())]
        rng.shuffle(queues)
        choices, parents = [], Counter()
        while queues and len(choices) < per_style:
            for queue in queues:
                # Prefer less-repeated parent documents within this stratum.
                best = min(range(len(queue)), key=lambda i: parents[queue[i].get('parent_id',queue[i]['doc_id'])])
                row = queue[best]; del queue[best]
                choices.append(row); parents[row.get('parent_id',row['doc_id'])] += 1
                if len(choices) == per_style: break
            queues = [q for q in queues if q]
        if len(choices) < per_style:
            raise ValueError(f'Not enough unique candidates for {style}: {len(choices)} < {per_style}')
        selected.extend(choices)
    rng.shuffle(selected)
    return selected


def build_packet(inputs, out, per_style=50, seed=20261006):
    if out.exists(): raise FileExistsError(out)
    if per_style < 1: raise ValueError('per-style must be positive')
    rows, by_content = [], defaultdict(list)
    for source in inputs:
        for line in source.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('style') not in STYLES: continue
            if not isinstance(row.get('text'),str) or not row['text'].strip(): continue
            row={**row,'input_file':str(source)}
            key=digest(' '.join(re.findall(r'\w+',row['text'].casefold())))
            by_content[key].append(row)
    conflicts=[]
    for key, group in by_content.items():
        if len({r['style'] for r in group}) > 1:
            conflicts.append({'text_hash':key,'doc_ids':[r['doc_id'] for r in group]})
        else: rows.append(group[0])
    selected=select_balanced(rows,per_style,seed)
    items, private = [], []
    for index,row in enumerate(selected):
        text,truncated=review_excerpt(row['text'])
        key=digest(f'{seed}:{index}:{row["doc_id"]}')[:20]
        items.append({'id':key,'text':text,'text_sha256':digest(text)})
        private.append({'id':key, 'candidate_doc_id':row['doc_id'],'provisional_style':row['style'],
                        'source_url':row.get('source_url',''),'source_domain':row.get('source_domain',''),
                        'genre':row.get('genre',''),'parent_id':row.get('parent_id',row['doc_id']),
                        'input_file':row['input_file'],'source_text_sha256':digest(row['text']),
                        'is_excerpt':truncated, 'words_shown':len(text.split())})
    if len({r['text_sha256'] for r in items}) != len(items):
        raise ValueError('Duplicate excerpts after truncation: revise candidate selection')
    packet={'schema_version':1,'purpose':'candidate_label_audit_not_external_test',
            'packet_sha256':packet_hash(items),'items':items}
    out.mkdir(parents=True)
    write_json(out/'packet.json',packet)
    write_jsonl(out/'private_provenance.jsonl',private)
    summary={'packet_sha256':packet['packet_sha256'],'seed':seed,'items':len(items),
             'inputs':{str(p):file_hash(p) for p in inputs},'sampling':'round_robin_source_genre_length',
             'provisional_counts':dict(Counter(r['provisional_style'] for r in private)),
             'source_counts':dict(Counter(r['source_domain'] for r in private)),
             'excerpt_count':sum(r['is_excerpt'] for r in private), 'conflicting_exact_duplicates':conflicts,
             'human_reviews_received':0, 'external_test':False,
             'limitation':'Source labels determine sampling strata only; packet may contain development/training documents.'}
    write_json(out/'manifest.json',summary)
    template=Path(__file__).with_name('annotation_template.html').read_text(encoding='utf-8')
    for slot in ('A','B'):
        payload=json.dumps({**packet,'labels':LABELS,'slot':slot},ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
        (out/f'reviewer_{slot}.html').write_text(template.replace('__PACKET_JSON__',payload),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True))
    return packet


def validate_review(packet, review):
    if review.get('schema_version') != 1 or review.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('Review belongs to a different packet/schema')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('Reviewer ID is required')
    if review.get('independent_review') is not True:
        raise ValueError('Independent review declaration is required')
    decisions=review.get('decisions',[])
    indexed={r['id']:r for r in decisions}
    if len(indexed) != len(decisions) or set(indexed) != {r['id'] for r in packet['items']}:
        raise ValueError('Exactly one decision per packet item is required')
    for row in packet['items']:
        decision=indexed[row['id']]
        if decision.get('text_sha256') != row['text_sha256']:
            raise ValueError('Reviewed text differs from packet')
        if decision.get('label') not in LABELS:
            raise ValueError('Unknown annotation label')
        if decision['label'] not in STYLES and not str(decision.get('reason','')).strip():
            raise ValueError('Non-style decisions require an explanation')
    return indexed


def compare_reviews(packet_path, review_a, review_b, out):
    if out.exists(): raise FileExistsError(out)
    packet=read_packet(packet_path)
    a=json.loads(review_a.read_text(encoding='utf-8')); b=json.loads(review_b.read_text(encoding='utf-8'))
    aa=validate_review(packet,a); bb=validate_review(packet,b)
    if a['reviewer'].strip().casefold() == b['reviewer'].strip().casefold():
        raise ValueError('Two distinct reviewer IDs are required')
    accepted, disagreements, other = [], [], []
    for item in packet['items']:
        ra,rb=aa[item['id']],bb[item['id']]
        row={**item,'review_a':ra,'review_b':rb}
        if ra['label'] != rb['label']: disagreements.append(row)
        elif ra['label'] in STYLES:
            accepted.append({**row,'style':ra['label'],'review_status':'two_reviews_agree_on_shown_text',
                             'usage_status':'not_automatically_added_to_training'})
        else: other.append(row)
    y=[aa[r['id']]['label'] for r in packet['items']]; z=[bb[r['id']]['label'] for r in packet['items']]
    kappa=None if len(set(y+z)) < 2 else float(cohen_kappa_score(y,z))
    summary={'packet_sha256':packet['packet_sha256'],'items':len(y),'reviewers':[a['reviewer'],b['reviewer']],
             'agreement':sum(x==v for x,v in zip(y,z))/len(y),'cohen_kappa_all_decisions':kappa,
             'kappa_scope':'all nine labels, including mixed/insufficient/non_kazakh/noise',
             'agreed_styles':len(accepted),'disagreements':len(disagreements),'agreed_nonstyle':len(other),
             'labels':list(LABELS),'confusion_matrix':confusion_matrix(y,z,labels=list(LABELS)).tolist(),
             'review_hashes':[file_hash(review_a),file_hash(review_b)],
             'limitation':'Reviewer identity, expertise and actual independence are self-declared, not verified by software.'}
    out.mkdir(parents=True)
    write_json(out/'summary.json',summary)
    write_jsonl(out/'agreed_styles.jsonl',accepted)
    write_jsonl(out/'needs_adjudication.jsonl',disagreements)
    write_jsonl(out/'agreed_nonstyle.jsonl',other)
    write_json(out/'adjudication_template.json',{
        'schema_version':1,'packet_sha256':packet['packet_sha256'],
        'review_hashes':summary['review_hashes'],'adjudicator':'','confirmed_read_shown_texts':False,
        'decisions':[{'id':r['id'],'text_sha256':r['text_sha256'],'label':None,'reason':''} for r in disagreements]})
    print(json.dumps(summary,ensure_ascii=True))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    build=commands.add_parser('build'); build.add_argument('--inputs',type=Path,nargs='+',required=True)
    build.add_argument('--out-dir',type=Path,required=True);build.add_argument('--per-style',type=int,default=50)
    build.add_argument('--seed',type=int,default=20261006)
    compare=commands.add_parser('compare');compare.add_argument('--packet',type=Path,required=True)
    compare.add_argument('--review-a',type=Path,required=True);compare.add_argument('--review-b',type=Path,required=True)
    compare.add_argument('--out-dir',type=Path,required=True)
    finalize=commands.add_parser('finalize');finalize.add_argument('--packet',type=Path,required=True)
    finalize.add_argument('--review-a',type=Path,required=True);finalize.add_argument('--review-b',type=Path,required=True)
    finalize.add_argument('--adjudication',type=Path);finalize.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='build':build_packet(args.inputs,args.out_dir,args.per_style,args.seed)
    elif args.action=='compare':compare_reviews(args.packet,args.review_a,args.review_b,args.out_dir)
    else:
        from kazstyle.data.adjudication import finalize_annotations
        finalize_annotations(args.packet,args.review_a,args.review_b,args.adjudication,args.out_dir)
