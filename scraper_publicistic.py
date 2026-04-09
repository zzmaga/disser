import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urldefrag, urljoin, urlparse

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
    )
}
DELAY = 1.0
SAVE_EVERY = 50
OUTPUT_FILE = "data/publicistic.csv"
LABEL = "publicistic"

SITES = [
    {
        "name": "abai",
        "base_url": "https://abai.kz",
        "listing_url_template": "https://abai.kz/part/69?page={page}",
        "start_page": 1,
        "max_pages": 150,
        "link_contains": "/post/",
        "text_selectors": [
            {"tag": "div", "attrs": {"class": "single-news"}},
        ],
        "min_text_length": 500,
        "prefer_paragraphs": False,
    },
    {
        "name": "egemen",
        "base_url": "https://egemen.kz",
        "listing_urls": [
            "https://egemen.kz/",
            "https://egemen.kz/articles",
        ],
        "link_contains": "/article/",
        "text_selectors": [
            {"tag": "article", "attrs": {"class": "white-block"}},
        ],
        "remove_selectors": [
            {"tag": "div", "attrs": {"class": "col-md-4"}},
            {"tag": "div", "attrs": {"class": "last-news-section -desktop"}},
            {"tag": "div", "attrs": {"class": "last-news"}},
        ],
        "min_text_length": 700,
        "prefer_paragraphs": True,
    },
]


def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=20)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(href, base_url):
    full_url = urljoin(base_url, href)
    full_url, _ = urldefrag(full_url)
    return full_url.rstrip("/")


def extract_year(soup):
    candidates = []

    time_tag = soup.find("time")
    if time_tag:
        if time_tag.get("datetime"):
            candidates.append(time_tag.get("datetime", ""))
        candidates.append(time_tag.get_text(" ", strip=True))

    meta_names = [
        ("property", "article:published_time"),
        ("property", "og:updated_time"),
        ("name", "pubdate"),
        ("name", "date"),
        ("itemprop", "datePublished"),
    ]
    for attr_name, attr_value in meta_names:
        meta = soup.find("meta", attrs={attr_name: attr_value})
        if meta and meta.get("content"):
            candidates.append(meta["content"])

    for value in candidates:
        match = re.search(r"(20\d{2})", value)
        if match:
            return int(match.group(1))

    return datetime.now().year


def select_blocks(soup, selector):
    return soup.find_all(selector["tag"], selector.get("attrs", {}))


def remove_unwanted(block, remove_selectors):
    for selector in remove_selectors:
        for bad in block.find_all(selector["tag"], selector.get("attrs", {})):
            bad.decompose()


def extract_text_from_block(block, prefer_paragraphs=False):
    paragraphs = []
    for p in block.find_all("p"):
        text = normalize_text(p.get_text(" ", strip=True))
        if len(text) >= 40:
            paragraphs.append(text)

    if prefer_paragraphs and len(paragraphs) >= 3:
        return " ".join(paragraphs)

    return normalize_text(block.get_text(" ", strip=True))


def cleanup_text(text, site_name):
    text = normalize_text(text)

    if site_name == "abai":
        text = re.sub(r"^\d+\s+\d+\s+пікір\s+", "", text)
        text = text.replace(" Screenshot ", " ")
        for marker in [
            " Ең көп оқылған ",
            " Үздік материалдар ",
            " Фото мұрағат ",
        ]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

    if site_name == "egemen":
        for marker in [
            " Соңғы жаңалықтар ",
            " Тағы да жүктеу ",
            " Оқи отырыңыз ",
        ]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

    return normalize_text(text)


def parse_text(url, site_config):
    r = fetch(url)
    if not r:
        return None, None

    soup = BeautifulSoup(r.text, "lxml")
    candidates = []

    for selector in site_config["text_selectors"]:
        for block in select_blocks(soup, selector):
            block_soup = BeautifulSoup(str(block), "lxml")
            if site_config.get("remove_selectors"):
                remove_unwanted(block_soup, site_config["remove_selectors"])

            text = extract_text_from_block(
                block_soup,
                prefer_paragraphs=site_config.get("prefer_paragraphs", False),
            )
            text = cleanup_text(text, site_config["name"])
            if len(text) >= site_config["min_text_length"]:
                candidates.append(text)

    if not candidates:
        return None, soup

    return max(candidates, key=len), soup


def get_article_links(listing_url, site_config):
    r = fetch(listing_url)
    if not r:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = normalize_url(href, site_config["base_url"])
        parsed = urlparse(full_url)
        site_domain = urlparse(site_config["base_url"]).netloc

        if parsed.netloc and parsed.netloc != site_domain:
            continue

        if site_config["link_contains"] not in parsed.path:
            continue

        links.add(full_url)

    return sorted(links)


def save_batch(batch, output_file):
    if not batch:
        return 0

    new_df = pd.DataFrame(batch)
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["source_url"], inplace=True)
    else:
        combined = new_df

    combined.to_csv(output_file, index=False, encoding="utf-8-sig")
    return len(combined)


def load_existing_urls(output_file):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Already collected: {len(df)} docs, resuming...")
        return set(df["source_url"].tolist())
    return set()


def iter_listing_urls(site_config):
    if "listing_urls" in site_config:
        for url in site_config["listing_urls"]:
            yield url
        return

    start_page = site_config.get("start_page", 1)
    max_pages = site_config.get("max_pages", 1)
    for page in range(start_page, max_pages + 1):
        yield site_config["listing_url_template"].format(page=page)


def scrape(max_docs=2000):
    os.makedirs("data", exist_ok=True)
    existing_urls = load_existing_urls(OUTPUT_FILE)
    collected = len(existing_urls)
    skipped = 0
    batch = []

    print(
        f"\nTarget: {max_docs} | Already have: {collected} | Left: {max_docs - collected}\n"
    )

    if collected >= max_docs:
        print("Target already reached.")
        return

    for site in SITES:
        if collected >= max_docs:
            break

        print(f"\n{'=' * 50}")
        print(f"Site: {site['name']}")

        empty_pages = 0
        listing_counter = 0

        for listing_url in iter_listing_urls(site):
            if collected >= max_docs:
                break

            listing_counter += 1
            print(f"  Listing {listing_counter}: {listing_url}")
            article_links = get_article_links(listing_url, site)

            if not article_links:
                empty_pages += 1
                print(f"  Empty listing ({empty_pages} in a row)")
                if empty_pages >= 3 and "listing_url_template" in site:
                    break
                time.sleep(DELAY)
                continue

            empty_pages = 0
            new_links = [url for url in article_links if url not in existing_urls]
            print(f"  Found: {len(article_links)} | New: {len(new_links)}")

            if not new_links:
                time.sleep(DELAY)
                continue

            for url in new_links:
                if collected >= max_docs:
                    break

                print(f"  [{collected + 1}/{max_docs}] ", end="", flush=True)
                text, soup = parse_text(url, site)

                if text:
                    batch.append(
                        {
                            "text": text,
                            "label": LABEL,
                            "source_url": url,
                            "year": extract_year(soup) if soup else datetime.now().year,
                            "site": site["name"],
                        }
                    )
                    existing_urls.add(url)
                    collected += 1
                    print(f"OK {len(text):,} chars - {url.split('/')[-1]}")
                else:
                    skipped += 1
                    print(f"SKIP - {url.split('/')[-1]}")

                if len(batch) >= SAVE_EVERY:
                    total = save_batch(batch, OUTPUT_FILE)
                    print(f"\nSaved {total} docs -> {OUTPUT_FILE}\n")
                    batch = []

                time.sleep(DELAY)

            time.sleep(DELAY)

    if batch:
        total = save_batch(batch, OUTPUT_FILE)
        print(f"\nFinal save: {total} docs")

    print(f"\n{'=' * 50}")
    print(f"Collected: {collected} | Skipped: {skipped} | {OUTPUT_FILE}")


if __name__ == "__main__":
    max_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    scrape(max_docs=max_docs)
