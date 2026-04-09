SOURCES = {
    "official": {
        "label": "official",
        "output_file": "data/official.csv",
        "rss_url": "https://adilet.zan.kz/kaz/docs/rss",
        "base_url": "https://adilet.zan.kz",
        "text_selector": {"tag": "div", "attrs": {"class": "text"}},
        "min_text_length": 500,
    }
    # сюда позже добавим научный, художественный, публицистический
}
