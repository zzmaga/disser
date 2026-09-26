"""Combine completed runs only when their data fingerprints match."""
import argparse
import json
from pathlib import Path
from kazstyle.settings import project_path

from kazstyle.data.corpus import load_manifest,write_json
from kazstyle.evaluation.reports import render_report


def compare(dataset,runs,out):
    frame,config=load_manifest(dataset)
    if out.exists():raise FileExistsError(out)
    results={}
    for run in runs:
        provenance=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        if provenance['manifest_sha256']!=config['manifest_sha256']:
            raise ValueError(f'Different dataset/split: {run}')
        items=json.loads((run/'results.json').read_text(encoding='utf-8'))
        if set(results)&set(items):raise ValueError('Duplicate model names')
        results.update(items)
    out.mkdir(parents=True)
    write_json(out/'results.json',results)
    write_json(out/'sources.json',{'runs':[str(p) for p in runs],'manifest_sha256':config['manifest_sha256']})
    render_report(out,frame,config,results,title='Сравнение моделей на трёх стилях')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=project_path('data/processed/pilot_v2'))
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,default=project_path('reports/pilot_v2'))
    a=p.parse_args();compare(a.dataset,a.runs,a.out_dir)
