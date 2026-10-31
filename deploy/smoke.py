"""Check a deployed preparation API using only synthetic input and stdlib."""
import json
import sys
import urllib.error
import urllib.request


def check(base):
    def request(path, data=None):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8') if data is not None else None
        req = urllib.request.Request(base.rstrip('/')+path, data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    health = request('/health')
    assert health['status'] == 'ready' and health['model_loaded'] is False
    result = request('/api/prepare', {'records': [
        {'id': 'a', 'text': '<p>Қазақ тілі.</p> https://example.org'},
        {'id': 'b', 'text': 'Қазақ тілі.'},
        {'id': 'c', 'text': 'https://example.net'}]})
    assert result['records'][0]['text'] == 'Қазақ тілі.'
    assert result['summary']['kept_rows'] == 1 and result['summary']['excluded_rows'] == 2
    try:
        request('/api/prepare', {'records': [{'id': 'x', 'text': 17}]})
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError('Invalid schema accepted')
    print(json.dumps({'status': 'passed', 'health': health, 'summary': result['summary'],
                      'schema_rejection': 422}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    check(sys.argv[1])
