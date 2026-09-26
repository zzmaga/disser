"""Create a read-only HTML/JSON audit of CSV structure and the completed pilot."""
import argparse
import html
import json
from pathlib import Path
from kazstyle.settings import project_path

import pandas as pd

from kazstyle.data.corpus import STYLES,load_manifest,read_csv_strict,write_json


def audit(data_dir,dataset,out):
    if out.exists():raise FileExistsError(out)
    audits={style:read_csv_strict(data_dir/f'{style}.csv')[1] for style in STYLES}
    frame,config=load_manifest(dataset)
    out.mkdir(parents=True)
    write_json(out/'raw_audit.json',audits)
    summary=json.loads((dataset/'summary.json').read_text(encoding='utf-8'))
    stats=pd.DataFrame([{'Стиль':style,'Документов':a['records'],'Физических строк':a['physical_lines'],
                         'Документов с переносами':a['embedded_newline_records'],
                         'Медиана слов':a['word_quantiles']['median'],'Максимум слов':a['word_quantiles']['max']}
                        for style,a in audits.items()])
    lengths=frame.groupby('style_name')[['excerpt_words','token_count']].agg(['min','median','max'])
    split=frame.groupby(['style_name','split']).size().unstack(fill_value=0)
    cross=pd.crosstab(frame.source_domain,frame.style_name)
    paragraphs=''
    for row in frame[frame.split=='train'].groupby('style_name').head(2).to_dict('records'):
        paragraphs+=f"<h3>{html.escape(row['style_name'])}</h3><p>{html.escape(row['text'])}</p><small>{html.escape(row['source_url'])}</small>"
    review=[json.loads(line) for line in (dataset/'annotation_sample_train.jsonl').read_text(encoding='utf-8').splitlines()]
    review_html=''.join(f"<details><summary>{html.escape(r['style'])} — {html.escape(r['doc_id'])}</summary>"
                        f"<p><a href='{html.escape(r['source_url'],quote=True)}'>Источник</a></p>"
                        f"<h4>Фрагмент модели</h4><p>{html.escape(r['text'])}</p>"
                        f"<h4>Полный очищенный документ</h4><p>{html.escape(r['text_clean'])}</p></details>" for r in review)
    header='''<!doctype html><html lang="ru"><meta charset="utf-8"><style>
body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#172b42}
table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #cbd5e1;padding:9px;text-align:right}
th{background:#eaf0f7}p{overflow-wrap:anywhere}details{border:1px solid #cbd5e1;padding:12px;margin:10px 0}
.note{padding:16px;background:#fff2cf}</style>'''
    (out/'report.html').write_text(header+f'''<title>Аудит корпуса</title><h1>Аудит корпуса и структуры CSV</h1>
<p>Записи CSV читаются целиком с учётом кавычек и переносов. Исходные файлы не изменены.</p>
{stats.to_html(index=False)}<h2>Пилот по трём стилям</h2>{split.to_html()}
<p>Равное количество документов каждого класса. Одна запись manifest = один фрагмент одного документа.</p>
<h2>Длина ввода моделей</h2>{lengths.to_html()}<p>Фрагментов, дополнительно ограниченных tokenizer-ом:
{summary['token_limited_examples']} из {summary['selected_documents']}. Модели получают одинаковый текст без скрытого усечения.</p>
<h2>Источники и метки</h2>{cross.to_html()}<p class="note">Разметка получена из источников и требует проверки.
Текущий дизайн не отделяет стиль от сайта. Новые источники необходимы для внешней оценки.</p>
<h2>Примеры фактического ввода из train</h2>{paragraphs}<h2>Ручная проверка 60 документов train</h2>
<p>Это материалы для проверки. Экспертные метки ещё не заполнены. Файл для заполнения: annotation_sample_train.jsonl.</p>
{review_html}</html>''',encoding='utf-8')
    print(out/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=project_path('data'))
    p.add_argument('--dataset',type=Path,default=project_path('data/processed/pilot_v2'))
    p.add_argument('--out-dir',type=Path,default=project_path('reports/data_audit_v2'))
    a=p.parse_args();audit(a.data_dir,a.dataset,a.out_dir)
