import tempfile
import unittest
from pathlib import Path

from kazstyle.data.corpus import digest,file_hash
from kazstyle.data.folktales import parse_folktale,repair_verified_source
from kazstyle.data.quality import clean_text


HTML='''<nav>Мәзір</nav><div itemprop="articleBody"><p>Бір
<button type="button" class="vocab-word" data-vocab-id="17">торғай</button>
қораға қонды.</p><button>Тыңдау</button><form>Кіру</form></div>
<aside>Басқа ертегі</aside>'''


class FolktaleTests(unittest.TestCase):
    def test_inline_words_survive_but_player_and_navigation_do_not(self):
        text,count=parse_folktale(HTML)
        self.assertEqual(clean_text(text)[0],'Бір торғай қораға қонды.')
        self.assertEqual(count,1)
        for noise in ['Мәзір','Тыңдау','Кіру','Басқа ертегі']:
            self.assertNotIn(noise,text)
        self.assertNotIn('торғай',parse_folktale(HTML,preserve_vocabulary=False)[0])

    def test_repair_requires_unchanged_source_legacy_text_and_new_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'page.html';path.write_text(HTML,encoding='utf-8')
            old=clean_text(parse_folktale(HTML,preserve_vocabulary=False)[0])[0]
            new=clean_text(parse_folktale(HTML)[0])[0]
            row={'source_domain':'ertegiler.kz','text':old}
            decision={'source_snapshot':str(path),'source_sha256':file_hash(path),
                      'text_sha256':digest(old),'repaired_text_sha256':digest(new),'inline_word_buttons':1}
            text,log=repair_verified_source(row,decision)
            self.assertEqual(text,new);self.assertEqual(log['restored_word_buttons'],1)
            for key,value in [('source_sha256','0'*64),('text_sha256','0'*64),
                              ('repaired_text_sha256','0'*64),('inline_word_buttons',2)]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    repair_verified_source(row,{**decision,key:value})
            with self.assertRaises(ValueError):repair_verified_source({**row,'text':'other'},decision)
            path.write_text('changed source',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):repair_verified_source(row,decision)


if __name__=='__main__':unittest.main()
