"""Export resolved, traceable annotation excerpts; never overwrite training data."""
import json
from collections import Counter
from pathlib import Path

from kazstyle.data.annotation import read_packet, validate_review, review_excerpt, LABELS
from kazstyle.data.corpus import STYLES, digest, file_hash, write_json, write_jsonl
from kazstyle.settings import PROJECT_ROOT


def finalize_annotations(packet_path,review_a,review_b,adjudication,out):
    if out.exists():raise FileExistsError(out)
    packet=read_packet(packet_path)
    a=json.loads(review_a.read_text(encoding='utf-8'));b=json.loads(review_b.read_text(encoding='utf-8'))
    aa=validate_review(packet,a);bb=validate_review(packet,b)
    if a['reviewer'].strip().casefold()==b['reviewer'].strip().casefold():
        raise ValueError('Two distinct reviewer IDs are required')
    disputed={key for key in aa if aa[key]['label']!=bb[key]['label']}
    hashes=[file_hash(review_a),file_hash(review_b)]
    resolved={};adjudicator=None
    if disputed and adjudication is None:raise ValueError('Unresolved disagreements: an adjudication file is required')
    if adjudication is not None:
        decision_file=json.loads(adjudication.read_text(encoding='utf-8'))
        if decision_file.get('schema_version')!=1 or decision_file.get('packet_sha256')!=packet['packet_sha256']:
            raise ValueError('Adjudication belongs to another packet')
        if decision_file.get('review_hashes')!=hashes:raise ValueError('Adjudication belongs to another review pair')
        adjudicator=decision_file.get('adjudicator')
        if not isinstance(adjudicator,str) or not adjudicator.strip() or decision_file.get('confirmed_read_shown_texts') is not True:
            raise ValueError('Adjudicator identity and reading declaration are required')
        decisions=decision_file.get('decisions',[]);resolved={r['id']:r for r in decisions}
        if len(resolved)!=len(decisions) or set(resolved)!=disputed:
            raise ValueError('Exactly one decision for every disagreement is required')
        items={r['id']:r for r in packet['items']}
        for key,row in resolved.items():
            if row.get('text_sha256')!=items[key]['text_sha256']:raise ValueError('Adjudicated text differs')
            if row.get('label') not in LABELS or not isinstance(row.get('reason'),str) or not row['reason'].strip():
                raise ValueError('Every adjudication requires a valid label and explanation')
    # Bind private provenance to the original frozen source files and shown text.
    folder=packet_path.parent
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['packet_sha256']!=packet['packet_sha256']:raise ValueError('Wrong private packet manifest')
    private=[json.loads(line) for line in (folder/'private_provenance.jsonl').read_text(encoding='utf-8').splitlines()]
    provenance={r['id']:r for r in private}
    if len(provenance)!=len(private) or set(provenance)!={r['id'] for r in packet['items']}:
        raise ValueError('Private provenance IDs differ from packet')
    sources={}
    for name,sha in manifest['inputs'].items():
        path=Path(name);path=path if path.is_absolute() else PROJECT_ROOT/path
        if file_hash(path)!=sha:raise ValueError('Original candidate snapshot changed')
        wanted={r['candidate_doc_id'] for r in private if r['input_file']==name}
        indexed={}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['doc_id'] in wanted:
                    if row['doc_id'] in indexed:raise ValueError('Duplicate source document ID')
                    indexed[row['doc_id']]=row
        if set(indexed)!=wanted:raise ValueError('Original source document missing')
        sources[name]=indexed
    accepted=[];excluded=[]
    for item in packet['items']:
        meta=provenance[item['id']]
        if meta['input_file'] not in sources:raise ValueError('Unregistered provenance input')
        source=sources[meta['input_file']][meta['candidate_doc_id']]
        shown,truncated=review_excerpt(source['text'])
        if digest(source['text'])!=meta['source_text_sha256'] or shown!=item['text'] or truncated!=meta['is_excerpt']:
            raise ValueError('Shown annotation does not match original source excerpt')
        if any(meta.get(key,'')!=source.get(key,'') for key in ['source_url','source_domain','genre']):
            raise ValueError('Private source metadata changed')
        final=resolved.get(item['id'],aa[item['id']]);label=final['label']
        record={**source,'doc_id':digest(packet['packet_sha256']+':'+item['id'])[:24],
            'source_document_id':source['doc_id'],'parent_id':source.get('parent_id') or source['doc_id'],
            'text':item['text'],'content_hash':digest(item['text'].casefold()),'style':label,
            'label_origin':'two_declared_human_reviews_with_adjudication_if_needed',
            'review_status':'adjudicated' if item['id'] in resolved else 'two_reviews_agree',
            'usage_status':'reviewed_excerpt_requires_dataset_and_split_protocol',
            'annotation':{'packet_sha256':packet['packet_sha256'],'item_id':item['id'],
                'text_sha256':item['text_sha256'],'review_hashes':hashes,'reviewer_ids':[a['reviewer'],b['reviewer']],
                'review_labels':[aa[item['id']]['label'],bb[item['id']]['label']],
                'adjudicator':adjudicator if item['id'] in resolved else None,
                'reason':final.get('reason',''),'original_provisional_style':source['style'],
                'label_scope':'Only the displayed excerpt; not the unseen full original document'}}
        # Original quality metrics may concern a longer text and are not reused.
        for key in ['quality','eligibility_reasons']:record.pop(key,None)
        (accepted if label in STYLES else excluded).append(record)
    out.mkdir(parents=True)
    write_jsonl(out/'reviewed_candidates.jsonl',accepted);write_jsonl(out/'excluded.jsonl',excluded)
    result={'packet_sha256':packet['packet_sha256'],'review_hashes':hashes,
        'adjudication_sha256':file_hash(adjudication) if adjudication else None,
        'accepted':len(accepted),'excluded':len(excluded),'resolved_disagreements':len(disputed),
        'style_counts':dict(Counter(r['style'] for r in accepted)),
        'training_modified':False,'limitation':'Human identity/expertise/independence are self-declared. Excerpts remain grouped with their source parent; no train/test assignment is made.'}
    write_json(out/'summary.json',result)
    print(json.dumps(result,ensure_ascii=True))
    return result
