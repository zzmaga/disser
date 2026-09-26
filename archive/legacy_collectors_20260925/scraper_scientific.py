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
OUTPUT_FILE = "data/scientific.csv"
LABEL = "scientific"

SITES = [
    {
        "name": "bilim_research",
        "base_url": "https://bilim-all.kz",
        "listing_url_template": "https://bilim-all.kz/article/list/51?page={page}",
        "start_page": 1,
        "max_pages": 50,
        "link_pattern": re.compile(r"^/article/\d+$"),
        "text_selectors": [
            {"tag": "div", "attrs": {"class": "article-content"}},
        ],
        "min_text_length": 800,
        "prefer_paragraphs": True,
    },
    {
        "name": "bilim_referats",
        "base_url": "https://bilim-all.kz",
        "listing_url_template": "https://bilim-all.kz/article/list/52?page={page}",
        "start_page": 1,
        "max_pages": 50,
        "link_pattern": re.compile(r"^/article/\d+$"),
        "text_selectors": [
            {"tag": "div", "attrs": {"class": "article-content"}},
        ],
        "min_text_length": 800,
        "prefer_paragraphs": True,
    },
]


SCIENTIFIC_MARKERS = [
    "аннотация",
    "abstract",
    "зерттеу",
    "зерттеуд",
    "ғылыми",
    "реферат",
    "курсов",
    "диплом",
    "әдістем",
    "гипотез",
    "литератур",
    "көзделген",
    "маqsат",
    "мақсат",
    "нәтиже",
    "қорытынды",
    "кіріспе",
    "тақырып",
    "университет",
    "факультет",
    "студент",
    "профессор",
    "доцент",
    "ф.ғ",
]
SCIENTIFIC_REJECT = [
    "туған күн",
    "құттықтау",
    "ертегі",
    "өлең",
    "тақpақ",
]


def is_scientific_text(text):
    lowered = text.lower()
    if any(marker in lowered for marker in SCIENTIFIC_REJECT):
        hits = sum(1 for marker in SCIENTIFIC_MARKERS if marker in lowered)
        if hits < 3:
            return False

    hits = sum(1 for marker in SCIENTIFIC_MARKERS if marker in lowered)
    return hits >= 2 or "аннотация" in lowered


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

    page_text = soup.get_text(" ", strip=True)
    candidates.append(page_text[:1000])

    for value in candidates:
        match = re.search(r"(20\d{2})", value)
        if match:
            return int(match.group(1))

    return datetime.now().year


def select_blocks(soup, selector):
    return soup.find_all(selector["tag"], selector.get("attrs", {}))


def extract_text_from_block(block, prefer_paragraphs=False):
    paragraphs = []
    for p in block.find_all("p"):
        text = normalize_text(p.get_text(" ", strip=True))
        if len(text) >= 40:
            paragraphs.append(text)

    if prefer_paragraphs and len(paragraphs) >= 3:
        return " ".join(paragraphs)

    return normalize_text(block.get_text(" ", strip=True))


def cleanup_text(text):
    return normalize_text(text)


def parse_text(url, site_config):
    r = fetch(url)
    if not r:
        return None, None

    soup = BeautifulSoup(r.text, "lxml")
    candidates = []

    for selector in site_config["text_selectors"]:
        for block in select_blocks(soup, selector):
            text = extract_text_from_block(
                block,
                prefer_paragraphs=site_config.get("prefer_paragraphs", False),
            )
            text = cleanup_text(text)
            if len(text) >= site_config["min_text_length"] and is_scientific_text(text):
                candidates.append(text)

    if not candidates:
        return None, soup

    return max(candidates, key=len), soup


def link_matches(href, site_config):
    parsed = urlparse(normalize_url(href, site_config["base_url"]))
    return bool(site_config["link_pattern"].match(parsed.path))


def get_article_links(listing_url, site_config):
    r = fetch(listing_url)
    if not r:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    links = set()
    site_domain = urlparse(site_config["base_url"]).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = normalize_url(href, site_config["base_url"])
        parsed = urlparse(full_url)

        if parsed.netloc and parsed.netloc != site_domain:
            continue

        if not link_matches(href, site_config):
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
        seen_on_site = set()

        for listing_url in iter_listing_urls(site):
            if collected >= max_docs:
                break

            listing_counter += 1
            print(f"  Listing {listing_counter}: {listing_url}")
            article_links = get_article_links(listing_url, site)

            if not article_links:
                empty_pages += 1
                print(f"  Empty listing ({empty_pages} in a row)")
                if empty_pages >= 3:
                    break
                time.sleep(DELAY)
                continue

            empty_pages = 0
            new_links = [
                url
                for url in article_links
                if url not in existing_urls and url not in seen_on_site
            ]
            seen_on_site.update(article_links)
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
