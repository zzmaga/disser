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

    def test_discretionary_breaks_do_not_turn_into_spaces_or_join_real_words(self):
        source='<h1>Story</h1><a href="/kz/news/literary?category=51">ӘҢГІМЕ</a><div class="content-text"><p>қой\u00adнына күн-түн екі сөз тар\u00ad\u00adтып</p></div>'
        text, meta = parse_story(source)
        self.assertEqual(text, 'қойнына күн-түн екі сөз тартып')
        self.assertEqual(meta['source_soft_hyphen_repairs'], 2)
        self.assertIn('\u00ad', parse_story(source, repair_soft_hyphens=False)[0])


if __name__=='__main__':unittest.main()
