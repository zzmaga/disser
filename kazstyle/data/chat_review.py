"""Screen longer messages for human review; never treat them as gold labels."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.quality import clean_text, inspect_text


def screen(source, existing, out, limit=150):
    if out.exists(): raise FileExistsError(out)
    if limit < 1: raise ValueError('limit must be positive')
    known={digest(json.loads(line)['text'].casefold()) for line in existing.read_text(encoding='utf-8').splitlines()}
    shortlist, counts = [], Counter()
    for index,line in enumerate(source.open(encoding='utf-8-sig')):
        counts['lines_scanned']+=1
        text=line.strip()
        if not 40 <= len(text.split()) <= 250: continue
        counts['length_40_to_250']+=1
        if len(re.findall('[әғқңөұүӘҒҚҢӨҰҮ]',text)) < 3: continue
        if re.search(r'https?://|www\.|@|\+7[\d ()-]{8,}|жарнама|промокод|казино|ставк',text,re.I): continue
        if re.search(r'^\d+[.)]|қымыз іш|барлығы ішілді|іше аласыз|ішу уақыты|админ құқық|Perplexity|\*\*|__|ассистент|нейрожелі|көмектесуге дайын|сұрақтарыңыз|айтып берейін бе',text,re.I): continue
        if not re.search(r'\b(?:мен|сен|маған|менің|менде|біз|біздің|сенің|саған|өзім|ғой|гой)\b',text,re.I): continue
        cleaned, changes=clean_text(text)
        key=digest(cleaned.casefold())
        if key in known: continue
        known.add(key)
        shortlist.append({'doc_id':digest(f'telegram_public:{index}')[:24], 'text':cleaned,
                          'source_row':index, 'cleaning':changes})
    # Hash order samples across the entire dump, never just its beginning.
    shortlist.sort(key=lambda r:r['doc_id'])
    accepted, rejected=[] ,[]
    for row in shortlist:
        stats,reasons=inspect_text(row['text'],min_words=35,max_words=250)
        record={**row,'style':'colloquial','genre':'long_message_candidate','quality':stats,
                'eligibility_reasons':reasons,'review_status':'unreviewed','usage_status':'review_pool_only',
                'label_origin':'personal_pronoun_screen_NOT_verified_style',
                'source_url':'https://huggingface.co/datasets/kurumikz/telegram-corpus-russian-kazakh',
                'source_domain':'telegram_public_dump','upstream_revision':'87fc5537b54c2133981bb33c7d3b6279145736bb',
                'parent_id':f'telegram_public_dump:block:{row["source_row"]//5000}',
                'parent_id_quality':'line_block_NOT_conversation_ID','license':'cc-by-nc-sa-4.0',
                'content_hash':digest(row['text'].casefold())}
        (rejected if reasons else accepted).append(record)
        if len(accepted)>=limit:break
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl',accepted);write_jsonl(out/'quarantine.jsonl',rejected)
    summary={'source_sha256':file_hash(source),'existing_sha256':file_hash(existing),'screening':dict(counts),
             'shortlisted':len(shortlist),'candidates':len(accepted),'quarantined':len(rejected),
             'expert_verified':False,'used_for_training':False,
             'limitations':['Pronoun and language filters bias selection; texts may be news, fiction, bots or quotations.',
                            'No true author/thread IDs; all labels require human review.']}
    write_json(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--existing',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=150)
    a=p.parse_args();screen(a.source,a.existing,a.out_dir,a.limit)
