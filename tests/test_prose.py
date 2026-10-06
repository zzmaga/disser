import unittest
from kazstyle.data.prose import parse_story


class ProseTests(unittest.TestCase):
    def test_story_body_excludes_navigation_and_separates_credit(self):
        source='''<h1>Әңгіме</h1><nav>Menu</nav><a href="/kz/news/literary?category=51">ӘҢГІМЕ</a>
        <div class="content-text"><p>Ол үйіне қайтты.</p><p>Тәржімалаған Біреу.</p></div><aside>Popular news</aside>'''
        text,meta=parse_story(source)
        self.assertEqual(text,'Ол үйіне қайтты.')
        self.assertEqual(meta['credits'],['Тәржімалаған Біреу.'])

    def test_ambiguous_prose_category_is_not_silently_literary(self):
        source='<h1>Opinion</h1><a href="/kz/news/literary?category=41">ПРОЗА</a><div class="content-text">Text</div>'
        with self.assertRaisesRegex(ValueError,'not tagged'):parse_story(source)


if __name__=='__main__':unittest.main()
