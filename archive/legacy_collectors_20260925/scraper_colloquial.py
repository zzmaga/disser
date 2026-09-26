"""
Collect Colloquial (разговорный) Kazakh texts from public Telegram channel previews.

Uses only public web pages (https://t.me/s/<channel>), no private API keys.
Filters for Kazakh-script content and informal length/style heuristics.
"""

import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "kk-KZ,kk;q=0.9,ru;q=0.5,en;q=0.3",
}
DELAY = 0.8
SAVE_EVERY = 50
OUTPUT_FILE = "data/colloquial.csv"
LABEL = "colloquial"

# Public channels with Kazakh informal / everyday language signal.
# Prefer lifestyle, language, statuses over hard news.
CHANNELS = [
    "status_kz",
    "qazaqtili",
    "otandastar",
    "massaget_kz",
    "jas_qazaq",
    "azattyq",
    "ertegiler_kz",
    "pogoda_astana",
]

KAZAKH_LETTERS = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")
CYRILLIC = re.compile(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")
URL_ONLY = re.compile(r"^https?://\S+$")
INFORMAL_MARKERS = re.compile(
    r"(?:\bғой\b|\bқой\b|\bекен\b|\bжарайды\b|\bкеремет\b|\bөте\b|"
    r"\bшын\b|\bбілемін\b|\bойлаймын\b|\bмаған\b|\bсаған\b|"
    r"\bдостар\b|\bбалалар\b|\bкүнде\b|\bқазір\b|\bкеше\b|"
    r"[😀-🙏❤️🔥😂😅😊🙂😉😍🥰😘👏🙏💯✨🎉])",
    re.IGNORECASE,
)


def fetch(session, url):
    try:
        r = session.get(url, timeout=20, allow_redirects=True)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None


def normalize_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def is_kazakh_colloquial(text):
    if not text or URL_ONLY.match(text):
        return False
    if len(text) < 40 or len(text) > 2500:
        return False

    words = text.split()
    if len(words) < 6 or len(words) > 350:
        return False

    kaz = len(KAZAKH_LETTERS.findall(text))
    cyr = len(CYRILLIC.findall(text))
    if cyr < 20:
        return False
    # Require some Kazakh-specific letters OR informal markers in Cyrillic text.
    if kaz < 2 and not INFORMAL_MARKERS.search(text):
        return False

    # Drop obvious boilerplate / channel mentions-only posts.
    if text.startswith("@") and len(words) < 8:
        return False
    if text.lower().startswith("channel created"):
        return False

    return True


def parse_messages(html, channel):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    oldest_id = None

    for msg in soup.select(".tgme_widget_message"):
        data_post = msg.get("data-post") or ""
        # data-post format: ChannelName/123
        post_id = None
        if "/" in data_post:
            post_id = data_post.split("/")[-1]
            try:
                oldest_id = int(post_id) if oldest_id is None else min(oldest_id, int(post_id))
            except ValueError:
                pass

        text_el = msg.select_one(".tgme_widget_message_text")
        if not text_el:
            continue
        text = normalize_text(text_el.get_text(" ", strip=True))
        if not is_kazakh_colloquial(text):
            continue

        date_el = msg.select_one("time")
        year = datetime.now().year
        if date_el and date_el.get("datetime"):
            m = re.search(r"(20\d{2})", date_el["datetime"])
            if m:
                year = int(m.group(1))

        source_url = (
            f"https://t.me/{channel}/{post_id}" if post_id else f"https://t.me/s/{channel}"
        )
        rows.append(
            {
                "text": text,
                "label": LABEL,
                "source_url": source_url,
                "year": year,
                "site": f"telegram_{channel}",
            }
        )

    more = soup.select_one("a.tme_messages_more, .js-messages_more, a[href*='before=']")
    before = None
    if more and more.get("href"):
        m = re.search(r"before=(\d+)", more["href"])
        if m:
            before = m.group(1)
    elif oldest_id is not None:
        before = str(oldest_id)

    return rows, before


def save_batch(batch, output_file):
    if not batch:
        return 0
    new_df = pd.DataFrame(batch)
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["source_url"], inplace=True)
        # also soft-dedup near-identical texts
        combined["text_norm"] = combined["text"].astype(str).str.lower().str.strip()
        combined.drop_duplicates(subset=["text_norm"], inplace=True)
        combined.drop(columns=["text_norm"], inplace=True)
    else:
        combined = new_df
    combined.to_csv(output_file, index=False, encoding="utf-8-sig")
    return len(combined)


def load_existing_urls(output_file):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Already collected: {len(df)} docs, resuming...")
        return set(df["source_url"].astype(str).tolist())
    return set()


def scrape_channel(session, channel, existing_urls, max_docs, collected, batch):
    before = None
    empty_pages = 0
    pages = 0
    max_pages = 80

    while collected < max_docs and pages < max_pages:
        url = f"https://t.me/s/{channel}"
        if before:
            url = f"{url}?before={before}"

        pages += 1
        print(f"  [{channel}] page {pages}: {url}")
        r = fetch(session, url)
        if not r or "tgme_widget_message" not in r.text:
            empty_pages += 1
            if empty_pages >= 2:
                break
            time.sleep(DELAY)
            continue

        rows, next_before = parse_messages(r.text, channel)
        new_rows = [row for row in rows if row["source_url"] not in existing_urls]

        if not new_rows:
            empty_pages += 1
            print(f"  no new colloquial rows ({empty_pages} empty)")
        else:
            empty_pages = 0
            for row in new_rows:
                if collected >= max_docs:
                    break
                batch.append(row)
                existing_urls.add(row["source_url"])
                collected += 1
                print(f"  [{collected}/{max_docs}] OK {len(row['text'])} chars - {row['source_url']}")
                if len(batch) >= SAVE_EVERY:
                    total = save_batch(batch, OUTPUT_FILE)
                    print(f"\nSaved {total} docs -> {OUTPUT_FILE}\n")
                    batch = []

        if not next_before or next_before == before:
            break
        before = next_before
        time.sleep(DELAY)

    return collected, batch


def scrape(max_docs=500):
    os.makedirs("data", exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    existing_urls = load_existing_urls(OUTPUT_FILE)
    collected = len(existing_urls)
    batch = []

    print(f"\nTarget: {max_docs} | Already have: {collected} | Left: {max_docs - collected}\n")
    if collected >= max_docs:
        print("Target already reached.")
        return

    for channel in CHANNELS:
        if collected >= max_docs:
            break
        print(f"\n{'=' * 50}")
        print(f"Channel: {channel}")
        collected, batch = scrape_channel(
            session, channel, existing_urls, max_docs, collected, batch
        )

    if batch:
        total = save_batch(batch, OUTPUT_FILE)
        print(f"\nFinal save: {total} docs")

    print(f"\n{'=' * 50}")
    print(f"Collected: {collected} | {OUTPUT_FILE}")


if __name__ == "__main__":
    max_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    scrape(max_docs=max_docs)
