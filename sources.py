SOURCES = {
    "official": {
        "label": "official",
        "output_file": "data/official.csv",
        "rss_url": "https://adilet.zan.kz/kaz/docs/rss",
        "base_url": "https://adilet.zan.kz",
        "text_selector": {"tag": "div", "attrs": {"class": "text"}},
        "min_text_length": 500,
    },
    "publicistic": {
        "label": "publicistic",
        "output_file": "data/publicistic.csv",
        "scraper": "scraper_publicistic.py",
    },
    "literary": {
        "label": "literary",
        "output_file": "data/literary.csv",
        "scraper": "scraper_literary.py",
    },
    "scientific": {
        "label": "scientific",
        "output_file": "data/scientific.csv",
        "scraper": "scraper_scientific.py",
    },
}
