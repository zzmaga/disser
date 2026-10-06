"""Run a frozen experiment plan sequentially with logs and explicit completion markers."""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from kazstyle.data.corpus import load_manifest,file_hash,write_json
from kazstyle.settings import PROJECT_ROOT, artifact_path


def code_hashes():
    paths=[PROJECT_ROOT/'manage.py',*sorted((PROJECT_ROOT/'kazstyle').rglob('*.py'))]
    return {p.relative_to(PROJECT_ROOT).as_posix():file_hash(p) for p in paths}


def build_jobs(plan):
    prefix=plan.get('run_prefix','text_only_v4')
    if not re.fullmatch(r'[a-z0-9_]+',prefix):raise ValueError('run_prefix must contain lowercase letters, digits or underscores')
    if any(not re.fullmatch(r'[a-z0-9_]+',m['id']) for m in plan['models']):raise ValueError('Invalid model ID in plan')
    jobs=[]
    for seed in plan['seeds']:
        suffix=f'{prefix}_classical_s{seed}'
        jobs.append({'id':suffix,'command':['train-classical','--dataset',plan['dataset'],'--out-dir','artifacts/'+suffix,'--seed',str(seed)]})
    for seed in plan['seeds']:
        for spec in plan['models']:
            suffix=f'{prefix}_{spec["id"]}_s{seed}'
            command=['train-transformer','--dataset',plan['dataset'],'--out-dir','artifacts/'+suffix,
                     '--model-name',spec['model_name'],'--head',spec['head'],'--epochs',str(plan['epochs']),
                     '--batch-size',str(spec['batch_size']),'--gradient-accumulation',str(spec['gradient_accumulation']),
                     '--lr',str(plan['learning_rate']),'--seed',str(seed),'--threads',str(plan['threads'])]
            if spec.get('revision'):command+=['--revision',spec['revision']]
            if spec['model_name'].startswith('google-bert/'):command+=['--allow-download']
            device=spec.get('device',plan.get('device','cpu'))
            precision=spec.get('precision',plan.get('precision','float32'))
            if device not in {'cpu','cuda','auto'} or precision not in {'float32','float16'}:
                raise ValueError('Invalid device or precision in plan')
            if device=='cpu' and precision=='float16':raise ValueError('FP16 plan requires CUDA')
            command+=['--device',device,'--precision',precision]
            if spec.get('gradient_checkpointing',plan.get('gradient_checkpointing',False)):
                command+=['--gradient-checkpointing']
            if spec.get('fused_adamw',plan.get('fused_adamw',False)):
                command+=['--fused-adamw']
            jobs.append({'id':suffix,'command':command})
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate planned run IDs')
    return jobs


def run(plan_path,out,resume=False,max_jobs=None):
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    _,config=load_manifest(PROJECT_ROOT/plan['dataset'])
    if config['manifest_sha256']!=plan['manifest_sha256']:raise ValueError('Plan dataset hash mismatch')
    if plan.get('dataset_config_sha256') and file_hash(PROJECT_ROOT/plan['dataset']/'config.json')!=plan['dataset_config_sha256']:
        raise ValueError('Plan dataset config hash mismatch')
    if max_jobs is not None and max_jobs<1:raise ValueError('max-jobs must be positive')
    jobs=build_jobs(plan)
    plan_hash=file_hash(plan_path)
    if resume:
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        state=json.loads((out/'status.json').read_text(encoding='utf-8'))
        if protocol['plan_sha256']!=plan_hash or protocol['plan']!=plan:
            raise ValueError('Cannot resume a changed plan')
        if protocol.get('source_sha256')!=code_hashes():raise ValueError('Cannot resume after source code changes')
        if protocol.get('python_executable')!=str(Path(sys.executable).resolve()):
            raise ValueError('Cannot resume in a different Python environment')
        if json.loads((out/'jobs.json').read_text(encoding='utf-8'))!=jobs:
            raise ValueError('Saved job commands differ')
        if state['status']=='complete':raise ValueError('Suite is already complete')
        if state['failed'] or state['current'] is not None:
            raise ValueError('Failed/interrupted run requires explicit recovery; never silently restart partial weights')
        expected=[j['id'] for j in jobs[:len(state['completed'])]]
        if state['completed']!=expected:raise ValueError('Completed jobs are not the planned prefix')
        for run_id in state['completed']:
            artifact=artifact_path(run_id)
            marker=json.loads((artifact/'COMPLETE.json').read_text(encoding='utf-8'))
            if marker['plan_sha256']!=plan_hash or marker['manifest_sha256']!=config['manifest_sha256']:
                raise ValueError('Completed run marker differs from plan')
            if file_hash(artifact/'results.json')!=marker.get('results_sha256'):
                raise ValueError('Completed results changed')
    else:
        if out.exists():raise FileExistsError(out)
        state={'status':'running','completed':[],'failed':[],'current':None,'total':len(jobs)}
    pending=jobs[len(state['completed']):]
    if any(artifact_path(j['id']).exists() for j in pending):
        raise FileExistsError('A pending run ID already exists in active storage or archive')
    if not resume:
        out.mkdir(parents=True)
        write_json(out/'protocol.json',{'plan':plan,'plan_sha256':plan_hash,
            'source_sha256':code_hashes(),'python_executable':str(Path(sys.executable).resolve()),
            'started_at':datetime.now(timezone.utc).isoformat()})
        write_json(out/'jobs.json',jobs)
    state['status']='running';launched=0
    for job in pending:
        state['current']=job['id'];write_json(out/'status.json',state)
        print('[suite] '+job['id'],flush=True)
        with (out/(job['id']+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8','manage.py',*job['command']],cwd=PROJECT_ROOT,
                                  stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['failed'].append({'id':job['id'],'returncode':result.returncode})
            state['status']='failed';write_json(out/'status.json',state)
            raise RuntimeError('Experiment failed; see log: '+job['id'])
        artifact=PROJECT_ROOT/'artifacts'/job['id']
        if not (artifact/'results.json').exists():raise RuntimeError('Run exited without results')
        write_json(artifact/'COMPLETE.json',{'plan_sha256':plan_hash,'manifest_sha256':config['manifest_sha256'],
                                          'results_sha256':file_hash(artifact/'results.json'),
                                          'finished_at':datetime.now(timezone.utc).isoformat()})
        state['completed'].append(job['id']);write_json(out/'status.json',state)
        launched+=1
        if max_jobs and launched>=max_jobs and len(state['completed'])<len(jobs):
            state.update(status='partial',current=None);write_json(out/'status.json',state)
            print('[suite] stopped at requested job boundary; resume preserves completed runs',flush=True)
            return state
    state['status']='complete';state['current']=None;write_json(out/'status.json',state)
    print('[suite] complete',flush=True)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-jobs',type=int)
    a=p.parse_args();run(a.plan,a.out_dir,a.resume,a.max_jobs)
