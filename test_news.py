# test_news2.py
import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "Mozilla/5.0"}

sites = [
    "https://tengrinews.kz/kazakhstan_news/",
    "https://www.kazpravda.kz/kaz/",
    "https://egemen.kz/",
    "https://abai.kz/",
]

for url in sites:
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        # ищем ссылки похожие на статьи (длинные URL)
        links = list(
            set(
                [
                    a["href"]
                    for a in soup.find_all("a", href=True)
                    if len(a["href"]) > 25
                    and not any(
                        x in a["href"]
                        for x in ["#", "javascript", ".jpg", ".png", "mailto"]
                    )
                ]
            )
        )
        print(f"✅ {url}")
        print(f"   Статус: {r.status_code} | Ссылок: {len(links)}")
        print(f"   Примеры: {links[:3]}")
    except Exception as e:
        print(f"❌ {url} — {e}")
    print()
