import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kazstyle.data.forum import BASE, collect

BODY = 'Мен жаңа телефон сатып алдым, бірақ оның қуаты өте тез таусылады. Өзім барлық баптауларын тексердім, бәрібір түсінбей отырмын. Сіздер осындай мәселені қалай шештіңіздер?'


class Response:
    status_code = 200

    def __init__(self, url, text):
        self.url, self.text = url, text

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.text.encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session:
    def __init__(self, pages, calls):
        self.pages, self.calls, self.headers = pages, calls, {}

    def get(self, url, **kwargs):
        self.calls.append(url)
        return Response(url, self.pages[url])

    def close(self):
        pass


class ForumCheckpointTests(unittest.TestCase):
    def test_interruption_after_download_replays_cached_post_without_losing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';calls=[];pages=self.pages()
            with patch('kazstyle.data.forum.requests.Session',side_effect=lambda:Session(pages,calls)), patch('kazstyle.data.forum.time.sleep'):
                with patch('kazstyle.data.forum.parse_question',side_effect=KeyboardInterrupt):
                    first=collect(out,2,2,['phones'])
                self.assertEqual(first['stop_reason'],'interrupted')
                self.assertEqual(first['candidates'],0)
                final=collect(out,2,2,['phones'],resume=True)
                self.assertTrue(final['collection_finished'])
                self.assertEqual(final['candidates'],2)
                self.assertEqual(calls.count(BASE+'100/'),1)

    def test_access_response_stops_job_and_prevents_automatic_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';response=Response(BASE+'robots.txt','denied')
            response.status_code=403
            with patch('kazstyle.data.forum.requests.Session') as factory:
                factory.return_value.get.return_value=response
                result=collect(out,2,2,['phones'])
                self.assertEqual(result['stop_reason'],'robots_access_or_rate_limit')
                self.assertFalse((out/'COMPLETE.json').exists())
                with self.assertRaisesRegex(ValueError,'do not automatically retry'):
                    collect(out,2,2,['phones'],resume=True)
                self.assertEqual(factory.return_value.get.call_count,1)

    def pages(self):
        return {BASE+'robots.txt':'User-agent: *\nAllow: /',
            BASE+'categories':'<a href="/questions/phones">Phones</a>',
            BASE+'questions/phones':'<div class="qa-q-item-title"><a href="/100/">one</a><a href="/200/">two</a></div>',
            **{BASE+str(i)+'/':f'<h1>Телефон туралы сұрақ {i}</h1><div class="qa-q-view-content"><div itemprop="text">{BODY}</div></div>' for i in [100,200]}}

    def test_budget_pause_resume_keeps_rows_and_does_not_fetch_first_post_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';calls=[];pages=self.pages()
            with patch('kazstyle.data.forum.requests.Session',side_effect=lambda:Session(pages,calls)), patch('kazstyle.data.forum.time.sleep'):
                first=collect(out,2,2,['phones'],max_requests=3)
                self.assertFalse(first['collection_finished'])
                self.assertEqual(first['stop_reason'],'request_budget')
                self.assertEqual(first['candidates'],1)
                first_row=(out/'candidates.jsonl').read_text(encoding='utf-8')
                self.assertFalse((out/'COMPLETE.json').exists())
                final=collect(out,2,2,['phones'],resume=True,max_requests=3)
                self.assertTrue(final['collection_finished'])
                self.assertEqual(final['candidates'],2)
                self.assertTrue((out/'candidates.jsonl').read_text(encoding='utf-8').startswith(first_row))
                self.assertEqual(calls.count(BASE+'100/'),1)
                self.assertEqual(calls.count(BASE+'200/'),1)
                self.assertTrue((out/'COMPLETE.json').exists())

    def test_stop_file_prevents_any_network_and_changed_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';calls=[];pages=self.pages()
            with patch('kazstyle.data.forum.requests.Session',side_effect=lambda:Session(pages,calls)), patch('kazstyle.data.forum.time.sleep'):
                collect(out,2,2,['phones'],max_requests=1)
                (out/'STOP').write_text('stop')
                before=len(calls)
                result=collect(out,2,2,['phones'],resume=True)
                self.assertEqual(result['stop_reason'],'stop_file')
                self.assertEqual(len(calls),before)
                with self.assertRaisesRegex(ValueError,'protocol'):
                    collect(out,3,2,['phones'],resume=True)

    def test_cached_page_tampering_is_not_silently_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';calls=[];pages=self.pages()
            with patch('kazstyle.data.forum.requests.Session',side_effect=lambda:Session(pages,calls)), patch('kazstyle.data.forum.time.sleep'):
                collect(out,2,2,['phones'],max_requests=2)
                state=json.loads((out/'CHECKPOINT.json').read_text(encoding='utf-8'))['state']
                path=out/'pages'/state['cache'][BASE+'questions/phones']['filename']
                path.write_text('tampered')
                # Restore a cursor needing the listing again, as after an interrupted listing fetch.
                from kazstyle.data.checkpoint import CollectionCheckpoint
                protocol=json.loads((out/'CHECKPOINT.json').read_text(encoding='utf-8'))['protocol']
                job=CollectionCheckpoint(out,protocol,{},True);job.state['pending']=[];job.save()
                with self.assertRaisesRegex(RuntimeError,'checksum'):
                    collect(out,2,2,['phones'],resume=True)


if __name__=='__main__':unittest.main()
