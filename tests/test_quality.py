import unittest

from kazstyle.data.quality import assert_text_only, clean_text, inspect_text


class QualityTests(unittest.TestCase):
    def test_multiline_photo_caption_is_removed_in_one_pass(self):
        raw='Қазақстан мен Ресей туралы хабар. Сурет\n:сайтынан\nалынды\nҚұжат көлік саласына қатысты.'
        cleaned,stats=clean_text(raw)
        self.assertEqual(cleaned,'Қазақстан мен Ресей туралы хабар. Құжат көлік саласына қатысты.')
        self.assertEqual(stats['photo_credits'],1)
        self.assertEqual(clean_text(cleaned)[0],cleaned)

    def test_addresses_removed_but_names_dates_and_style_survive(self):
        original='Өтініш\nБ. Қалиевке 04.10.2026 күні сұраймын. abai.kz https://x.org/a?q=1 www.a.kz a@b.kz @account'
        text,stats=clean_text(original)
        self.assertEqual(text,'Өтініш Б. Қалиевке 04.10.2026 күні сұраймын.')
        self.assertEqual(stats['addresses'],4)
        assert_text_only([text])

    def test_url_changes_do_not_change_model_text(self):
        base='Қазақстанда ғылым мен білім туралы хабар жарияланды.'
        self.assertEqual(clean_text(base+' https://abai.kz/1')[0],clean_text(base+' https://adilet.zan.kz/2')[0])
        for value in ['abai.kz','www.example.org','user@example.com','@channel']:
            with self.assertRaises(ValueError):assert_text_only(['Қазақша мәтін '+value])

    def test_web_header_removed_without_stripping_form_header(self):
        text,_=clean_text('Жаңалықтар 1267 0 пікір 11 Наурыз, 2026 сағат 12:50 Білім туралы. Abai.kz Сурет: ernur.kz сайтынан алынды.')
        self.assertEqual(text,'Білім туралы.')
        self.assertIn('Өтініш',clean_text('Өтініш Мектеп директоры Б. Қалиевке')[0])

    def test_html_scripts_and_contacts(self):
        text,_=clean_text('<nav>Menu</nav><p>Қазақша <b>мәтін</b> 😊</p><script>secret()</script> Тел: 8747-12345678')
        self.assertEqual(text,'Қазақша мәтін 😊')

    def test_foreign_language_and_numeric_noise_rejected(self):
        for value in ['Это обычный русский текст о городе, его жителях и новых событиях.',
                      'Я не маю доступу до ваших особистих повідомлень, але можу відповісти на ваше запитання.',
                      '123 456 789 '*50]:
            _,reasons=inspect_text(value,min_words=5)
            self.assertTrue(reasons)


if __name__=='__main__':unittest.main()
