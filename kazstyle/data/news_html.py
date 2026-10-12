"""Extract article prose from saved Egemen HTML without recommendation paragraphs."""
import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from kazstyle.data.news_cleanup import revise_text

VERSION = 'egemen_article_prose_v2'


def parse_article(raw, url):
    soup = BeautifulSoup(raw, 'html.parser')
    bodies = soup.select('.article-prose')
    if len(bodies) != 1:
        raise ValueError('Expected exactly one article-prose container')
    body, removed = bodies[0], []
    for node in body.select('script,style,nav,aside,figure,figcaption,form,button'):
        if node.parent is None:
            continue
        removed.append({'kind':'nonarticle_element', 'text':node.get_text(' ', strip=True)})
        node.decompose()
    # Some blockquotes contain BOTH a real quotation and a recommendation.
    # Inspect the individual paragraphs first, preserving the quotation itself.
    for node in list(body.select('p')) + list(body.select('blockquote')):
        if node.parent is None:
            continue
        links = node.select('a[href]')
        if (links and node.get_text(' ', strip=True) == ' '.join(a.get_text(' ', strip=True) for a in links)
                and all(urlsplit(urljoin(url, a['href'])).netloc in {'egemen.kz','www.egemen.kz'}
                        and '/article/' in urlsplit(urljoin(url, a['href'])).path
                        and urljoin(url, a['href']).split('#')[0] != url.split('#')[0] for a in links)):
            removed.append({'kind':'related_article_links', 'text':node.get_text(' ', strip=True),
                            'urls':[urljoin(url,a['href']) for a in links]})
            node.decompose()
    for node in list(body.select('p')):
        value = node.get_text(' ', strip=True)
        if len(value.split()) <= 25 and re.match(r'^(?:Фото\s*:|Сурет(?:тер)?\s*:|Суретті (?:түсірген|салған)|Коллажды жасаған|Коллаж\s*:)', value, re.I):
            removed.append({'kind':'photo_credit_paragraph', 'text':value}); node.decompose()
    # Preserve adjacency across inline tags (e.g. Сәтпаев<a>...</a>қа).
    # Only block boundaries introduce whitespace; get_text(' ') splits words.
    for br in body.select('br'):
        br.replace_with('\n')
    for node in list(body.select('p,div,li,h1,h2,h3,h4,blockquote,tr')):
        node.insert_before('\n'); node.insert_after('\n')
    raw_text = body.get_text('', strip=False)
    if len(raw_text.split()) < 8:
        raise ValueError('Article body too short after extraction')
    metadata = {'contains_table': body.find('table') is not None}
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            obj = json.loads(node.string or '')
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get('@type') == 'NewsArticle':
            metadata.update({key:obj.get(key) for key in ['headline','datePublished','dateModified','author','inLanguage']})
            break
    text, cleaning = revise_text(raw_text, 'egemen.kz')
    return text, metadata, removed, cleaning
