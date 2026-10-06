import unittest
from kazstyle.data.forum import parse_question, has_contact_identifier


class ForumExtractionTests(unittest.TestCase):
    def test_contact_filter_handles_domestic_prefix_and_attached_words(self):
        for text in ['нөмір87001234567осы', '+7 (700) 123-45-67', 'ЖСН123456789012бар']:
            self.assertTrue(has_contact_identifier(text), text)
        for text in ['2005-2007 жылдар', 'бағасы 200000 теңге', '04.10.2026', '1234567890123456']:
            self.assertFalse(has_contact_identifier(text), text)

    def page(self,title,body):
        return '<html><h1>'+title+'</h1><nav>Мәзір</nav><div class="qa-q-view-content"><div itemprop="text">'+body+'</div></div><div class="qa-a-item-content">Басқа адамның жауабы</div></html>'

    def test_punctuation_does_not_duplicate_question_header(self):
        text,author=parse_question(self.page('Телефон жоғалды. Қалай табамын?','Телефон жоғалды.Қалай табамын?'))
        self.assertEqual(text,'Телефон жоғалды.Қалай табамын?')
        self.assertIsNone(author)
        self.assertNotIn('Басқа адамның жауабы',text)

    def test_word_fragment_comments_are_removed(self):
        text,_=parse_question(self.page('Көмек керек','&lt;!--StartFragment--&gt;Көмек керек &lt;!--[if !supportLists]--&gt; мәтін &lt;!--EndFragment--&gt;'))
        self.assertNotIn('<!--',text)
        self.assertNotIn('supportLists',text)
        self.assertIn('мәтін',text)


if __name__=='__main__':unittest.main()
