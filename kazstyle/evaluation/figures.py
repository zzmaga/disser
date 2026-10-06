"""Export research figures from completed, verified experiment artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kazstyle.data.corpus import load_manifest,file_hash,write_json
from kazstyle.settings import PROJECT_ROOT, artifact_path, evidence_path

NAMES={'word_tfidf_logreg':'Word TF-IDF + LR','word_tfidf_linear_svc':'Word TF-IDF + SVM',
       'char_tfidf_linear_svc':'Character TF-IDF + SVM','mbert_last':'mBERT',
       'kaz_roberta_last':'Kaz-RoBERTa (last)','kaz_roberta_concat4':'Kaz-RoBERTa (concat4)'}


def build(summary,out):
    if out.exists():raise FileExistsError(out)
    evidence=json.loads((summary/'provenance.json').read_text(encoding='utf-8'))
    suite=evidence_path(evidence['suite'])
    plan=json.loads((suite/'protocol.json').read_text(encoding='utf-8'))['plan']
    frame,config=load_manifest(PROJECT_ROOT/plan['dataset'])
    if config['manifest_sha256']!=evidence['manifest_sha256']:raise ValueError('Wrong figure dataset')
    for name,sha in evidence['prediction_hashes'].items():
        if file_hash(evidence_path(name))!=sha:raise ValueError('Saved predictions changed')
    results=json.loads((summary/'results.json').read_text(encoding='utf-8'))
    bootstrap=json.loads((summary/'bootstrap.json').read_text(encoding='utf-8'))
    test=frame[frame.split=='test'];test_n=len(test)
    max_support=int(test.groupby('label').size().max())
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    labels=list(NAMES);positions=np.arange(len(labels))
    means=np.array([results[n]['test_macro_f1_mean'] for n in labels])
    std=np.array([results[n]['test_macro_f1_sample_std'] for n in labels])
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained')
    ax.barh(positions,means,xerr=std,color='#2b658b',capsize=4,height=.62)
    ax.set_yticks(positions,list(NAMES.values()));ax.invert_yaxis();ax.set_xlim(0,1.08)
    ax.set_xticks(np.linspace(0,1,6));ax.set_xlabel('Macro-F1 — mean ± sample SD across seeds 42, 43, 44')
    ax.set_title(f'Development test: {test_n} texts, provisional labels',loc='left',pad=16)
    for y,mean,sd in zip(positions,means,std):ax.text(mean+sd+.015,y,f'{mean:.3f}',va='center',fontsize=10)
    ax.set_axisbelow(True);ax.xaxis.grid(alpha=.2);save(fig,'macro_f1_three_seeds')
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained')
    bounds=np.array([bootstrap['models'][n]['paired_macro_f1_difference_vs_reference_95_interval'] for n in labels])
    observed=np.array([next(r['test_macro_f1'] for r in results[n]['runs'] if r['seed']==42)-
                       next(r['test_macro_f1'] for r in results['char_tfidf_linear_svc']['runs'] if r['seed']==42) for n in labels])
    # Draw endpoints directly: no assumption that a percentile interval contains the observed statistic.
    ax.hlines(positions,bounds[:,0],bounds[:,1],color='#2b658b',linewidth=3)
    ax.scatter(observed,positions,color='#143f5c',zorder=3);ax.axvline(0,color='#a84436',linestyle='--',linewidth=1)
    ax.set_yticks(positions,list(NAMES.values()));ax.invert_yaxis();ax.set_xlabel('Macro-F1 difference versus character SVM')
    ax.set_title('Seed 42: paired group bootstrap, 95% conditional intervals',loc='left',pad=16)
    ax.text(0,-.24,'2000 draws within classes; development set only. No multiple-comparison correction.',transform=ax.transAxes,fontsize=9)
    ax.xaxis.grid(alpha=.2);save(fig,'paired_bootstrap_difference')
    style_names=[config['id_to_label'][str(i)] for i in range(5)]
    fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
    for ax,name in zip(axes.flat,labels):
        representative=next(r for r in results[name]['runs'] if r['seed']==42)
        detail=json.loads((artifact_path(representative['run'])/'results.json').read_text(encoding='utf-8'))[name]
        matrix=np.array(detail['test']['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max_support)
        ax.set_title(NAMES[name]);ax.set_xticks(range(5),style_names,rotation=35,ha='right')
        ax.set_yticks(range(5),style_names);ax.set_xlabel('Predicted');ax.set_ylabel('Reference label')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>max_support/2 else '#17394d')
    fig.suptitle(f'Development-test confusion matrices — fixed seed 42, {test_n} texts',fontsize=15)
    save(fig,'confusion_matrices_seed42')
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    distributions=[frame.loc[frame.label==i,'excerpt_words'].to_numpy() for i in range(5)]
    ax.boxplot(distributions,tick_labels=style_names,showfliers=True,patch_artist=True,
               boxprops={'facecolor':'#b9d5e5','edgecolor':'#2b658b'},medianprops={'color':'#a84436','linewidth':2})
    ax.set_ylabel('Words in the actual model input');ax.set_title(f'Input-length distributions across all {len(frame)} texts',loc='left',pad=16)
    ax.set_ylim(0,max(map(np.max,distributions))+15);ax.yaxis.grid(alpha=.2);ax.set_axisbelow(True)
    save(fig,'input_lengths')
    write_json(out/'provenance.json',{'manifest_sha256':config['manifest_sha256'],'summary_sha256':file_hash(summary/'results.json'),
        'bootstrap_sha256':file_hash(summary/'bootstrap.json'),'matplotlib':matplotlib.__version__,
        'interpretation':'Development-set figures with provisional labels; not an external quality claim.'})
    print(json.dumps({'figures':len(list(out.glob('*.png'))),'out':str(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();build(a.summary,a.out_dir)
