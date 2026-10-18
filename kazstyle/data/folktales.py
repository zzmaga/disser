"""Recover inline vocabulary words from verified Ertegiler HTML snapshots.

This repairs extraction, not spelling or style labels. Frozen corpus versions
remain unchanged; decisions are consumed only by a new export recipe.
"""
from pathlib import Path

from bs4 import BeautifulSoup

from kazstyle.data.corpus import digest, file_hash
from kazstyle.data.quality import clean_text


def extract_body(body, *, preserve_vocabulary=True):
    # Work on a copy: extracting legacy and repaired text must use the same DOM.
    body=BeautifulSoup(str(body),'html.parser')
    vocabulary=body.select('button.vocab-word[data-vocab-id]')
    count=len(vocabulary)
    if preserve_vocabulary:
        for node in vocabulary:
            node.unwrap()
    for node in body.select('script,style,nav,footer,form,button'):
        node.decompose()
    return body.get_text('\n',strip=True),count


def parse_folktale(raw, *, preserve_vocabulary=True):
    soup=BeautifulSoup(raw,'html.parser')
    body=soup.select_one('[itemprop="articleBody"]')
    if body is None:
        raise ValueError('Folktale articleBody missing')
    return extract_body(body,preserve_vocabulary=preserve_vocabulary)


def repair_verified_source(row, decision):
    if row.get('source_domain')!='ertegiler.kz':
        raise ValueError('Unsupported source for vocabulary repair')
    snapshot=Path(decision['source_snapshot'])
    if file_hash(snapshot)!=decision['source_sha256']:
        raise ValueError('Folktale snapshot changed')
    raw=snapshot.read_bytes()
    legacy,_=parse_folktale(raw,preserve_vocabulary=False)
    repaired,count=parse_folktale(raw)
    legacy=clean_text(legacy)[0];repaired=clean_text(repaired)[0]
    if legacy!=row['text'] or digest(legacy)!=decision['text_sha256']:
        raise ValueError('Historical folktale extraction does not match the candidate')
    if count!=decision['inline_word_buttons'] or count<1:
        raise ValueError('Inline vocabulary button count changed')
    if digest(repaired)!=decision['repaired_text_sha256'] or repaired==legacy:
        raise ValueError('Repaired folktale differs from the reviewed recipe')
    return repaired,{'rule':'source_verified_vocabulary_buttons','restored_word_buttons':count,
                     'source_sha256':decision['source_sha256'],
                     'repaired_text_sha256':decision['repaired_text_sha256']}
