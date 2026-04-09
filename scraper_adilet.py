# scraper_adilet.py — финальная рабочая версия

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import urllib3
from sources import SOURCES

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 1.0
SAVE_EVERY = 50
BASE = "https://adilet.zan.kz"
PAGE_SIZE = 100  # максимум документов на странице каталога


def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None


def get_doc_links_from_catalog_page(year, page_num):
    """Получает ссылки на документы со страницы каталога за год."""
    url = f"{BASE}/kaz/search/docs/dt={year}-&page={page_num}&pagesize={PAGE_SIZE}"
    r = fetch(url)
    if not r:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/kaz/docs/" in href and len(href) > 12:
            if not any(
                x in href
                for x in ["/rss", "/compare", "/comments", "/help", "/search", "dt="]
            ):
                full = BASE + href if href.startswith("/") else href
                links.append(full)

    return list(set(links))


def parse_text(url, selector, min_length):
    r = fetch(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    block = soup.find(selector["tag"], selector.get("attrs", {}))
    if block:
        text = block.get_text(separator=" ", strip=True)
        if len(text) >= min_length:
            return text
    return None


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
        print(f"Уже собрано: {len(df)} документов — продолжаем")
        return set(df["source_url"].tolist())
    return set()


def scrape(source_config, max_docs=2000):
    label = source_config["label"]
    output_file = source_config["output_file"]
    selector = source_config["text_selector"]
    min_length = source_config["min_text_length"]

    os.makedirs("data", exist_ok=True)
    existing_urls = load_existing_urls(output_file)
    collected = len(existing_urls)

    print(
        f"\nЦель: {max_docs} | Уже есть: {collected} | Осталось: {max_docs - collected}\n"
    )

    if collected >= max_docs:
        print("✅ Цель уже достигнута!")
        return

    batch = []
    skipped = 0
    years = list(range(2026, 1999, -1))

    for year in years:
        if collected >= max_docs:
            break

        print(f"\n{'='*50}")
        print(f"📆 Год: {year}")
        page = 1
        empty_pages = 0

        while collected < max_docs:
            print(f"  📄 Страница {page} ({PAGE_SIZE} докум./стр.)")
            doc_links = get_doc_links_from_catalog_page(year, page)

            if not doc_links:
                empty_pages += 1
                print(f"  Пустая страница ({empty_pages} подряд)")
                if empty_pages >= 3:
                    break
                page += 1
                time.sleep(DELAY)
                continue

            empty_pages = 0
            new_links = [l for l in doc_links if l not in existing_urls]
            print(f"  Найдено: {len(doc_links)} | Новых: {len(new_links)}")

            if not new_links:
                # все ссылки уже собраны — переходим к следующей странице
                page += 1
                time.sleep(DELAY)
                continue

            for url in new_links:
                if collected >= max_docs:
                    break

                print(f"  [{collected+1}/{max_docs}] ", end="", flush=True)
                text = parse_text(url, selector, min_length)

                if text:
                    batch.append(
                        {"text": text, "label": label, "source_url": url, "year": year}
                    )
                    existing_urls.add(url)
                    collected += 1
                    print(f"✅ {len(text):,} симв. — {url.split('/')[-1]}")
                else:
                    skipped += 1
                    print(f"⚠️  пропущено — {url.split('/')[-1]}")

                # сохраняем каждые SAVE_EVERY документов
                if len(batch) >= SAVE_EVERY:
                    total = save_batch(batch, output_file)
                    print(f"\n💾 Сохранено {total} документов → {output_file}\n")
                    batch = []

                time.sleep(DELAY)

            page += 1
            time.sleep(DELAY)

    # сохраняем остаток
    if batch:
        total = save_batch(batch, output_file)
        print(f"\n💾 Финальное сохранение: {total} документов")

    print(f"\n{'='*50}")
    print(f"✅ Собрано: {collected} | ⚠️ Пропущено: {skipped} | 💾 {output_file}")


if __name__ == "__main__":
    config = SOURCES["official"]
    scrape(config, max_docs=2000)
