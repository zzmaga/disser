import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kazstyle.data.poetry import parse_poem
from kazstyle.data.legal_import import screen_text,build as build_legal
from kazstyle.data.forum_posts import parse_posts,screen_post,build as build_posts
from kazstyle.data.corpus import file_hash

BODY='Мен жаңа телефон сатып алдым, бірақ оның қуаты тез таусылады. Өзім баптауларын тексердім, бәрібір түсінбей отырмын. Сіздер осындай мәселені қалай шештіңіздер?'
LEGAL='Қазақстан Республикасының Үкіметі ҚАУЛЫ ЕТЕДІ. Осы қаулыға сәйкес мемлекеттік мекеменің жұмысын ұйымдастыру тәртібі бекітілсін. Қаулының орындалуын бақылау облыс әкімінің орынбасарына жүктелсін. Құжат алғашқы ресми жарияланған күнінен кейін күшіне енеді. Мекеме басшысы қызметкерлерді бекітілген тәртіппен таныстыруды қамтамасыз етсін.'

class ExpansionTests(unittest.TestCase):
    def test_reserved_thread_rejected_by_legacy_post_extractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'input.jsonl'
            source.write_text(json.dumps({'usage_status':'reserved_final_test'})+'\n')
            with self.assertRaisesRegex(ValueError, 'Reserved'):
                build_posts([source], root/'out')
            self.assertFalse((root/'out').exists())

    def test_poem_only_body_and_explicit_author(self):
        page='<h1 class="heading">Күз</h1><p>Пікірлер 100</p><div class="poem-content">Бірінші жол<br>Екінші жол<br>Үшінші жол<br>Төртінші жол</div><div class="olen-author-box"><figure><h5><a href="/figure/12">Ақын Аты</a></h5></figure></div>'
        text,meta=parse_poem(page,'https://bilim-all.kz/olen/12-kuz')
        self.assertNotIn('Пікірлер',text);self.assertNotIn('Ақын Аты',text)
        self.assertEqual(meta['verse_lines'],4);self.assertEqual(meta['author_name'],'Ақын Аты')
        with self.assertRaises(ValueError):parse_poem(page.replace('olen-author-box','unrelated'),'https://bilim-all.kz/olen/12-kuz')

    def test_legal_known_furniture_removed_without_dropping_legal_body(self):
        raw='Правительство Республики Казахстан (Премьер-Министр РК) '+LEGAL+' ҚР НҚА ЭББ-гі нақты құжатқа сілтеу QR-коды'
        text,_,reasons,_=screen_text(raw)
        self.assertEqual(text,LEGAL);self.assertEqual(reasons,[])
        self.assertIn('residual_bilingual_pdf_furniture',screen_text('Маслихат г. Петропавловска '+LEGAL)[2])

    def test_municipal_header_scope_preserves_quotes_body_and_unknown_names(self):
        header='Акимат Костанайской области'
        prefix='Қостанай облысының әкімдігі '
        cleaned,_,reasons,log=screen_text(prefix+header+' '+LEGAL)
        self.assertNotIn(header,cleaned)
        self.assertIn(LEGAL,cleaned)
        self.assertIn(header,log['removed_publisher_furniture'])
        self.assertEqual(reasons,[])
        for raw in [prefix+'«'+header+'» '+LEGAL,
                    prefix+LEGAL*3+' '+header,
                    'Акимат Неизвестной области '+prefix+LEGAL]:
            text,_,reasons,_=screen_text(raw)
            self.assertIn('Акимат',text)
            self.assertIn('residual_bilingual_pdf_furniture',reasons)

    def test_invalid_upstream_record_logged_not_invented_or_silently_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            raw=(json.dumps({'id':5,'src':'zan.gov.kz','kk':LEGAL},ensure_ascii=False)+'\n').encode()+b'{"kk":"\xd0'
            (source/'upstream.jsonl.gz').write_bytes(gzip.compress(raw))
            (source/'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'revision':'fixed','raw_sha256':hashlib.sha256(raw).hexdigest()}))
            result=build_legal(source,root/'out',10)
            self.assertEqual(result['selected'],1);self.assertEqual(len(result['invalid_records']),1)
            self.assertEqual(result['invalid_records'][0]['line'],2)
            self.assertEqual(gzip.decompress((source/'upstream.jsonl.gz').read_bytes()),raw)
            with self.assertRaises(FileExistsError):build_legal(source,root/'out',10)

    def forum_page(self):
        return f'''<h1>Телефон туралы сұрақ</h1><div class="qa-q-view"><div class="qa-q-view-content"><div itemprop="text">{BODY}</div></div><time itemprop="dateCreated" datetime="2016-10-06T17:09:36+0000"></time></div>
        <div class="qa-a-list-item" id="a44"><p>50 дауыс</p><div class="qa-a-item-content"><div itemprop="text">Мен де солай ойлаймын. {BODY}</div></div><time itemprop="dateCreated" datetime="2017-01-01T10:00:00+0000"></time><div class="qa-a-item-who-data"><a class="qa-user-link" href="../user/name">name</a></div></div>'''

    def test_answers_are_individual_posts_grouped_with_their_thread(self):
        posts=parse_posts(self.forum_page(),'https://surak.baribar.kz/123/')
        self.assertEqual(len(posts),2);self.assertNotEqual(posts[0]['doc_id'],posts[1]['doc_id'])
        self.assertEqual(posts[0]['parent_id'],posts[1]['parent_id'])
        self.assertEqual(posts[1]['source_url'],'https://surak.baribar.kz/123/#a44')
        self.assertNotIn('50 дауыс',posts[1]['text']);self.assertNotIn('name',posts[1]['text'])
        self.assertEqual(screen_post(posts[1])[2],[])

    def test_dates_and_quoted_homework_do_not_become_verified_conversation(self):
        base={'text':BODY,'created_at':'2017-01-01T10:00:00+0000'}
        for stamp in [None,'2026-01-01T10:00:00+0000']:
            self.assertIn('missing_or_recent_source_date',screen_post({**base,'created_at':stamp})[2])
        self.assertIn('recent_source_edit',screen_post({**base,'modified_at':'2025-01-01'})[2])
        self.assertIn('homework_or_formal_quoted_material',screen_post({**base,'text':BODY+' Маған эссе керек.'})[2])

    def test_rejected_copy_does_not_discard_valid_dated_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = root/'page.html'
            # Both answers have exactly the same words; only the second is eligible.
            raw = self.forum_page()
            answer = raw[raw.index('<div class="qa-a-list-item"'):]
            raw = raw.replace('2017-01-01', '2025-01-01') + answer.replace('id="a44"', 'id="a45"')
            page.write_text(raw, encoding='utf-8')
            source = root/'input.jsonl'
            source.write_text(json.dumps({'source_domain':'surak.baribar.kz',
                'source_url':'https://surak.baribar.kz/123/', 'source_snapshot':str(page),
                'source_sha256':file_hash(page)})+'\n', encoding='utf-8')
            summary = build_posts([source], root/'out', existing_ids=['known-old-id'])
            accepted = [json.loads(line) for line in (root/'out/candidates.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['answer_id'] for r in accepted if r['genre']=='forum_answer'], ['45'])
            self.assertEqual(json.loads((root/'out/excluded_existing_ids.json').read_text()), ['known-old-id'])
            self.assertEqual(summary['cleanup_version'], 'forum_utterance_screen_v3')

    def test_conflicting_snapshot_hashes_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input.jsonl'
            rows = [{'source_domain':'surak.baribar.kz', 'source_snapshot':'same.html', 'source_sha256':sha}
                    for sha in ['first', 'second']]
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Conflicting hashes'):
                build_posts([source], root/'out')
            self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main()
