import unittest
import json
from kazstyle.data.business import parse_document, listing_links
from bs4 import BeautifulSoup
from kazstyle.data.quality import clean_text


class BusinessExtractionTests(unittest.TestCase):
    def test_ajax_pagination_html_is_decoded(self):
        payload = json.dumps({'html': '<div class="doc-item"><div class="name"><a href="/documents/35">form</a></div></div>', 'next': '?page=3'})
        links, following = listing_links(payload, 'https://resmihat.kz/documents/category/3?page=2')
        self.assertEqual(links, ['https://resmihat.kz/documents/35'])
        self.assertEqual(following, 'https://resmihat.kz/documents/category/3?page=3')

    def test_observed_load_more_attribute_followed_not_navigation_links(self):
        page = BeautifulSoup('<nav><a href="/documents/999">menu</a></nav><div class="doc-item"><div class="name"><a href="/documents/34">form</a></div></div><a next-page-url="?page=2">more</a>', 'html.parser')
        links, next_url = listing_links(page, 'https://resmihat.kz/documents/category/3')
        self.assertEqual(links, ['https://resmihat.kz/documents/34'])
        self.assertEqual(next_url, 'https://resmihat.kz/documents/category/3?page=2')

    def test_only_kazakh_body_and_inline_words_preserved(self):
        page = '<h1>Document</h1><nav>MENU</nav><div id="integr_content_kz"><p>Е<span>ңбек</span> шарты</p><p>Бірінші тармақ.</p><table><tr><td>Екінші</td><td>Үшінші</td></tr></table></div><div id="integr_content_ru">Русский перевод</div>'
        text, title = parse_document(page)
        cleaned, _ = clean_text(text)
        self.assertEqual(cleaned, 'Еңбек шарты Бірінші тармақ. Екінші Үшінші')
        self.assertEqual(title, 'Document')

    def test_no_fallback_to_navigation_when_document_absent(self):
        with self.assertRaises(ValueError):
            parse_document('<h1>Title</h1><p>Navigation and other text</p>')


if __name__ == '__main__':
    unittest.main()
