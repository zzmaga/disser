"""Versioned, label-independent text cleaning shared by training and inference.

Source addresses belong in provenance, never in model inputs. This module does
not guess style labels or remove meaningful names, dates, punctuation or emojis.
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

CLEANING_VERSION = 'text_only_v4'
ADDRESS = re.compile(
    r'https?://[^\s<>]+|www\.[^\s<>]+|'
    r'(?<![\w@])[\w.+-]+@[\w.-]+\.[a-z]{2,24}\b|'
    r'(?<![\w@])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+'
    r'(?:kz|қаз|com|org|net|ru|edu|gov|io|me|info|рф|news|dev|app|online|site|biz|ai)(?:/[^\s<>]*)?', re.I)
HANDLE = re.compile(r'(?<!\w)@[\w]{3,}')
PHOTO = re.compile(r'(?:Сурет|Фото)(?:тің авторы)?\s*:\s*[^\n]{0,160}?(?:сайтынан алынды|алынды|жеке мұрағатынан)[.!?]?', re.I)
NEWS_COUNTER = re.compile(r'^.{0,100}?\b\d+\s+\d+\s+пікір\s+\d{1,2}\s+[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+,?\s+20\d{2}\s+сағат\s+\d{1,2}:\d{2}\s*', re.I)
FOOTER = re.compile(r'(?:Пікір (?:қалдыру|жазу)|Пікіріңізді жазыңыз|Осы материалды бөлісіңіз|Читайте также|Біздің арнаға жазылыңыз)\s*[:!]?.*$', re.I | re.S)


def clean_text(value: str) -> tuple[str, dict]:
    if not isinstance(value, str):
        raise TypeError('text must be a string')
    text = unicodedata.normalize('NFC', html.unescape(value))
    changes = {}
    if re.search(r'</?[a-z][^>]*>', text, re.I):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'noscript']):
            element.decompose()
        text = soup.get_text(' ')
        changes['html_removed'] = True
    for name, pattern in [('photo_credits', PHOTO), ('news_header', NEWS_COUNTER),
                          ('footer', FOOTER), ('addresses', ADDRESS), ('handles', HANDLE)]:
        text, count = pattern.subn(' ', text)
        if count:
            changes[name] = count
    text,count=re.subn(r'\b(?:тел(?:ефон)?\.?|телефон нөмірі)\s*:?\s*\+?[\d ()-]{7,}', ' ', text, flags=re.I)
    if count:changes['phone_contacts']=count
    text=re.sub(r'_{2,}', ' ', text)
    # Zero-width/control noise, not Kazakh letters or style-bearing punctuation.
    text = ''.join(' ' if unicodedata.category(c) in {'Cc', 'Cf'} else c for c in text)
    text = re.sub(r'\s+', ' ', text).strip()
    # URL removal and newline normalization can expose a formerly broken caption.
    # Remove it in the same pass, so the website does not clean it differently.
    text,count=PHOTO.subn(' ',text)
    if count:
        changes['photo_credits']=changes.get('photo_credits',0)+count
        text=re.sub(r'\s+', ' ',text).strip()
    return text, changes


def assert_text_only(texts):
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty/non-string model input at row {index}')
        if ADDRESS.search(text) or HANDLE.search(text):
            raise ValueError(f'Address or account handle leaked into model input at row {index}')


@lru_cache(maxsize=1)
def language_detector():
    from lingua import Language, LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_languages(Language.KAZAKH, Language.RUSSIAN, Language.ENGLISH,
        Language.UKRAINIAN, Language.BELARUSIAN, Language.TURKISH, Language.AZERBAIJANI).with_low_accuracy_mode().build()


def inspect_text(text, min_words=12, max_words=100_000):
    from lingua import Language
    words = text.split()
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    stats = {'words': len(words), 'letter_fraction': letters/max(1,len(text)),
             'digit_fraction': digits/max(1,len(text)),
             'unique_word_fraction': len(set(w.casefold() for w in words))/max(1,len(words))}
    reasons = []
    if not min_words <= len(words) <= max_words:
        reasons.append('length')
    if stats['letter_fraction'] < .5 or stats['digit_fraction'] > .15:
        reasons.append('table_or_nonprose')
    if len(words) >= 40 and stats['unique_word_fraction'] < .20:
        reasons.append('repetition')
    if '\ufffd' in text or re.search(r'Р[\x80-\xbf]', text):
        reasons.append('encoding_damage')
    # A language score is a screening signal, not a verified annotation.
    values = language_detector().compute_language_confidence_values(text[:12000]) if letters else []
    stats['kazakh_score'] = next((v.value for v in values if v.language == Language.KAZAKH), 0.)
    if stats['kazakh_score'] < .65:
        reasons.append('language_review')
    if ADDRESS.search(text) or HANDLE.search(text):
        reasons.append('address_remaining')
    return stats, reasons
