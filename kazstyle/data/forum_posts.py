"""Extract source-dated forum utterances from saved pages, with thread grouping.

The deterministic screen supplies provisional candidates, never human labels.
Only pre-2023 posts are admitted to reduce recent generated-answer contamination.
"""
from collections import Counter
from datetime import datetime
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urljoin,urlsplit

from bs4 import BeautifulSoup
from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.forum import parse_question,has_contact_identifier,post_timestamps
from kazstyle.data.quality import clean_text,inspect_text

VERSION='forum_utterance_screen_v3'
PERSONAL=re.compile(r'\b(?:мен|менің|менде|маған|мені|өзім|сен|сенің|саған|сіз|сіздер|сендер|біз|біздің|ғой|гой|сәлем|салем|достар|рахмет|рақмет|айтыңыздаршы|айтындаршы|көмектесіңіздерші|көмектесіндерші|білесіздер|білесіңдер|ойлаймын)\b',re.I)
MIXED=re.compile(r'\b(?:реферат|эссе|шығарма|теңдеу|теорема|бұйрық|қаулы|мазмұндама|презентация|тақырыбында|тақырыбына|сочинение|доклад|дипломдық|диссертация)\b|\b[xyх]\s*[=²³]|\b\d+\s*[+*/=]\s*\d+',re.I)
AUTOMATION=re.compile(r'chatgpt|нейрожелі|жасанды интеллект|ассистент|көмектесуге дайын|сұрақтарыңыз|промокод|казино|ставк|жарнама|\*\*|```',re.I)
ADDRESS=re.compile(r'\b(?:сізге|сізді|өзің|өзіңнің|өзіңіз|көріңіз|көрмейсің|нөміріме|нөмірім|телефоным|компьютерім|машинам|досым|түсінбедім|білмеймін|ойлайсың|ойлайсыз|жіберіңіз|айтасыздар|беріңіздер|менікі)\b',re.I)
STRONG_PERSONAL=re.compile(r'\b(?:мен|менің|менде|маған|мені|өзім|ғой|гой|достар|сәлем|салем|рахмет|рақмет|ойлаймын|түсінбедім|білмеймін)\b',re.I)

def timestamp(node):
    return post_timestamps(node)

def parse_posts(raw,url):
    soup=BeautifulSoup(raw,'html.parser');match=re.search(r'/(\d+)/?',urlsplit(url).path)
    if not match or urlsplit(url).netloc!='surak.baribar.kz':raise ValueError('Not a source thread URL')
    thread=match[1];question=soup.select_one('.qa-q-view')
    if question is None:raise ValueError('Question container missing')
    text,author=parse_question(raw);created,modified=timestamp(question)
    posts=[{'doc_id':digest('surak_question:'+thread)[:24],'text':text,'genre':'forum_question',
        'source_url':url,'created_at':created,'modified_at':modified,'author_group':author}]
    for answer in soup.select('.qa-a-list-item[id]'):
        ident=re.fullmatch(r'a(\d+)',answer['id']);body=answer.select_one('.qa-a-item-content [itemprop="text"]')
        if not ident or body is None:continue
        for node in body.select('script,style,nav,form,button'):node.decompose()
        text=re.sub(r'<!--.*?-->',' ',html.unescape(body.get_text(' ',strip=True)),flags=re.S)
        byline=answer.select_one('.qa-a-item-who-data .qa-user-link[href]')
        author=digest('surak_author:'+urljoin(url,byline['href']))[:24] if byline else None
        created,modified=timestamp(answer)
        posts.append({'doc_id':digest('surak_answer:'+ident[1])[:24],'text':text,'genre':'forum_answer',
            'source_url':url.split('#')[0]+'#'+answer['id'],'created_at':created,'modified_at':modified,
            'author_group':author,'answer_id':ident[1]})
    return [{**p,'parent_id':'surak_thread:'+thread,'source_domain':'surak.baribar.kz'} for p in posts]

def screen_post(post):
    text,cleaning=clean_text(post['text']);stats,reasons=inspect_text(text,min_words=8,max_words=350)
    created=post.get('created_at');modified=post.get('modified_at')
    try:
        if not created or datetime.fromisoformat(created).year>=2023:reasons.append('missing_or_recent_source_date')
        if modified and datetime.fromisoformat(modified).year>=2023:reasons.append('recent_source_edit')
    except ValueError:reasons.append('invalid_source_date')
    if not (PERSONAL.search(text) or ADDRESS.search(text)):reasons.append('no_explicit_interpersonal_language')
    if post.get('genre')=='forum_answer' and len(text.split())>110 and not STRONG_PERSONAL.search(text):
        reasons.append('long_impersonal_answer_requires_review')
    if MIXED.search(text):reasons.append('homework_or_formal_quoted_material')
    if AUTOMATION.search(text):reasons.append('automation_ad_or_markup_signal')
    if has_contact_identifier(text):reasons.append('possible_contact_or_identifier')
    if text.count('«')+text.count('»')+text.count('"')>=8:reasons.append('quote_heavy_message')
    return text,stats,sorted(set(reasons)),cleaning

def build(input_paths,out,existing_ids=()):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    known=set(existing_ids);pages={};input_hashes={}
    for name in input_paths:
        path=Path(name);input_hashes[path.as_posix()]=file_hash(path)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line)
                if str(r.get('usage_status', '')).startswith('reserved'):
                    raise ValueError('Reserved question threads cannot enter a development post pool')
                if r.get('source_domain')=='surak.baribar.kz' and r.get('source_snapshot') and r.get('source_sha256'):
                    previous = pages.setdefault(r['source_snapshot'], r)
                    if previous['source_sha256'] != r['source_sha256']:
                        raise ValueError('Conflicting hashes for the same forum snapshot')
    accepted=[];rejected=[];errors=[];ids=set();hashes=set()
    for i,(snapshot,r) in enumerate(pages.items(),1):
        if file_hash(snapshot)!=r['source_sha256']:raise ValueError('Forum source snapshot changed')
        try:posts=parse_posts(Path(snapshot).read_bytes(),r['source_url'])
        except ValueError as e:errors.append({'snapshot':snapshot,'error':str(e)});continue
        for post in posts:
            if post['doc_id'] in ids or post['doc_id'] in known:continue
            ids.add(post['doc_id']);text,stats,reasons,cleaning=screen_post(post)
            sha=digest(' '.join(text.casefold().split()))
            if sha in hashes:reasons.append('normalized_exact_duplicate')
            # A rejected undated/recent copy must not suppress a valid older post.
            if not reasons:hashes.add(sha)
            row={**post,'text':text,'style':'colloquial','source_snapshot':snapshot,'source_sha256':r['source_sha256'],
                 'quality':stats,'cleaning':cleaning,'eligibility_reasons':reasons,'cleanup_version':VERSION,
                 'review_status':'rule_screened_NOT_human_review','usage_status':'review_pool_only',
                 'label_origin':'forum_genre_plus_predeclared_rules_NOT_expert',
                 'license':'Publisher and author rights apply; no blanket redistribution permission asserted'}
            (rejected if reasons else accepted).append(row)
        if i%500==0:print('Forum pages',i,'eligible utterances',len(accepted),flush=True)
    for p,sha in input_hashes.items():
        if file_hash(p)!=sha:raise ValueError('Collection input changed during screening; wait for collector completion')
    out.mkdir(parents=True);write_jsonl(out/'candidates.jsonl',accepted)
    write_json(out/'excluded_existing_ids.json', sorted(known))
    write_jsonl(out/'audit.jsonl',[{k:v for k,v in r.items() if k not in {'text','cleaning'}} for r in rejected])
    summary={'source_pages':len(pages),'candidates':len(accepted),'by_genre':dict(Counter(r['genre'] for r in accepted)),
        'quarantined':len(rejected),'errors':errors,'reasons':dict(Counter(v for r in rejected for v in r['eligibility_reasons'])),
        'input_hashes':input_hashes,'excluded_existing_ids_sha256':file_hash(out/'excluded_existing_ids.json'),
        'cleanup_version':VERSION,'expert_verified':False,'used_for_training':False,
        'limitations':['Rules and source dates are not proof of human authorship or verified style.',
            'Question/answer posts are separate natural utterances, always grouped by their original thread.',
            'Interpersonal-word and date filters bias the sample; evaluate length, genre and source transfer separately.']}
    write_json(out/'collection.json',summary);shutil.copy2(__file__,out/'code_snapshot.py')
    write_json(out/'COMPLETE.json', {p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()})
    return summary
