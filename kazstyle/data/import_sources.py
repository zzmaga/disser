"""Screen downloaded news/chat; never equate every Telegram post with conversation."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from kazstyle.data.corpus import digest, write_json, write_jsonl
from kazstyle.data.quality import clean_text, inspect_text
from kazstyle.settings import project_path


def convert(out, chat_limit=3000):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    accepted=[];reasons=Counter();seen=set()
    news=project_path('data/external/news_20261002/kaz_news_corpus_clean.jsonl')
    with news.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);raw=row['content'];url=row['url']
            if raw.count('• Кеше')>=2 or raw.count('• Бүгін')>=3:
                reasons['news_sidebar_instead_of_article']+=1;continue
            text,changes=clean_text(raw);stats,problems=inspect_text(text)
            key=digest(text.casefold())
            if key in seen:problems.append('duplicate')
            if problems:reasons.update(problems);continue
            seen.add(key)
            accepted.append({'doc_id':digest(url)[:24],'text':text,'style':'publicistic','genre':'news',
                'parent_id':url,'source_url':url,'source_domain':urlsplit(url).netloc,
                'label_origin':'publisher_article','review_status':'unreviewed','upstream':'kurumikz/kaz-news-corpus',
                'license':'odc-by','quality':stats,'cleaning':changes})
    print(f'[import] clean news={len(accepted)}',flush=True)
    chat=project_path('data/external/chat_20261002/telegram_data.txt')
    pool=[]
    with chat.open(encoding='utf-8-sig') as stream:
        for index,line in enumerate(stream):
            raw=line.strip()
            # No short exclamations, long forwarded articles, contact dumps or ads.
            if not 8<=len(raw.split())<=50 or len(re.findall('[әғқңөұүӘҒҚҢӨҰҮ]',raw))<2:continue
            if re.search(r'https?://|www\.|@|\+7\d|жарнама|промокод|ставк|казино',raw,re.I):
                reasons['chat_forward_or_ad']+=1;continue
            if re.search(r'^\d+[.)]|қымыз іш|барлығы ішілді|іше аласыз|ішу уақыты|админ құқық|Perplexity|\*\*|__',raw,re.I):
                reasons['chat_bot_quiz_or_markup']+=1;continue
            if not re.search(r'\b(?:мен|сен|менің|маған|менде|мені|біз|бізде|біздің|сенің|саған|сені|сендер|өзім|ғой|гой|қойш|бауырым)\b',raw,re.I):
                reasons['chat_style_ambiguous']+=1;continue
            if re.search(r'сұрақтарыңыз|көмектесуге дайын|айтып берейін бе|сұрақтар бар ма|ассистент|нейрожелі',raw,re.I):
                reasons['chat_assistant_response']+=1;continue
            text,changes=clean_text(raw);stats,problems=inspect_text(text,min_words=8,max_words=80)
            key=digest(text.casefold())
            if key in seen:continue
            if problems:reasons.update(problems);continue
            seen.add(key)
            pool.append({'doc_id':digest(f'telegram_public:{index}')[:24],'text':text,'style':'colloquial',
                'genre':'informal_message','source_url':'https://huggingface.co/datasets/kurumikz/telegram-corpus-russian-kazakh',
                'source_domain':'telegram_public_dump','parent_id':f'telegram_public_dump:block:{index//5000}',
                'source_row':index,'label_origin':'informal_corpus_screen','review_status':'unreviewed',
                'license':'cc-by-nc-sa-4.0','parent_id_quality':'conservative_line_block_NOT_conversation_ID',
                'quality':stats,'cleaning':changes})
            if len(pool)%2000==0:print(f'[import] chat candidates={len(pool)} scanned={index}',flush=True)
    # Hash-ranked sample across the whole file, not the first available messages.
    pool=sorted(pool,key=lambda r:r['doc_id'])[:chat_limit]
    accepted.extend(pool)
    write_jsonl(out/'candidates.jsonl',accepted)
    write_json(out/'summary.json',{'accepted_by_style':dict(Counter(r['style'] for r in accepted)),
        'screened_out_reasons':dict(reasons),'chat_limit':chat_limit,
        'limitations':['No expert style labels.','Chat conversation and author IDs are absent; block grouping is only a proxy.',
                       'Colloquial examples are short; length-stratified evaluation is required.']})
    print(f'[import] news/chat candidates={len(accepted)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--chat-limit',type=int,default=3000)
    a=p.parse_args();convert(a.out_dir,a.chat_limit)
