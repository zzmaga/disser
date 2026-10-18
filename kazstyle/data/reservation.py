"""Prospective evaluation reservations, distinct from approved reference labels.

The compact ledger contains identities, hashes and group aliases, never model
features or predictions. New training checks the registered ledgers by default;
reading/restoring historical experiments does not apply a new embargo to them.
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from urllib.parse import urlsplit

from kazstyle.data.corpus import STYLES, digest, file_hash, normalized_hash, write_json, write_jsonl
from kazstyle.data.grouping import GROUP_KEYS, group_values
from kazstyle.settings import project_path

VERSION = 'prospective_reservation_v1'
REGISTRY = project_path('configs/evaluation_reservations.json')
IDENTITY_KEYS = ('doc_id', 'source_url', *GROUP_KEYS, 'split_group_id')
HASH_KEYS = ('text_sha256', 'normalized_text_sha256', 'corpus_text_sha256',
             'view_parent_text_sha256', 'parent_content_hash', 'excerpt_hash')


def individual_source_url(row):
    url = row.get('source_url')
    if not isinstance(url, str):
        return None
    host = (urlsplit(url).hostname or '').removeprefix('www.')
    publisher = str(row.get('source_domain') or '').removeprefix('www.')
    if host in {'huggingface.co', 'github.com', 'raw.githubusercontent.com'}:
        return None
    return url if host and host == publisher else None


def identities(row):
    result = set()
    for key in IDENTITY_KEYS:
        if key == 'source_url' and individual_source_url(row) is None:
            continue  # A dataset/archive download URL is not a document ID.
        for value in group_values(row, key):
            # Both names refer to the canonical full-document component.
            result.add(('split_group_id' if key == 'corpus_split_group_id' else key, value))
    for key in HASH_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            result.add(('text_hash', value))
    text = row.get('text')
    if isinstance(text, str):
        result.update([('text_hash', digest(text)), ('text_hash', normalized_hash(text))])
    aliases = row.get('duplicate_occurrences')
    for alias in aliases if isinstance(aliases, list) else []:
        for key in ('doc_id', 'source_url', 'text_sha256', 'normalized_text_sha256'):
            if key == 'source_url' and individual_source_url(alias) is None:
                continue
            if isinstance(alias.get(key), str):
                result.add(('text_hash' if key.endswith('sha256') else key, alias[key]))
    return result


def source_matches(row, selector):
    return all(row.get(key) == value for key, value in selector.items())


def _index(rows):
    result = defaultdict(set)
    for row in rows:
        for key, value in identities(row):
            result[key].add(value)
    return {key: sorted(values) for key, values in sorted(result.items())}


def registered_ledgers(registry=None):
    registry = Path(registry) if registry is not None else REGISTRY
    if not registry.exists():
        raise FileNotFoundError(registry)
    spec = json.loads(registry.read_text(encoding='utf-8'))
    loaded = []
    for entry in spec['ledgers']:
        path = project_path(entry['path'])
        if file_hash(path) != entry['sha256']:
            raise ValueError('Evaluation reservation ledger changed after registration')
        ledger = json.loads(path.read_text(encoding='utf-8'))
        loaded.append((path, ledger))
    return loaded


def protected_rows(rows, ledgers=None):
    """Return reasons, including source rules and aliases after ID/view changes."""
    rows = list(rows)
    reasons = defaultdict(set)
    for _, ledger in registered_ledgers() if ledgers is None else ledgers:
        index = {key: set(values) for key, values in ledger['protected_identities'].items()}
        for i, row in enumerate(rows):
            if any(value in index.get(key, ()) for key, value in identities(row)):
                reasons[i].add(ledger['reservation_id'] + ':identity_or_group')
            if any(source_matches(row, selector) for selector in ledger['source_selectors']):
                reasons[i].add(ledger['reservation_id'] + ':withheld_source')
    for i, row in enumerate(rows):
        if str(row.get('usage_status', '')).startswith('reserved'):
            reasons[i].add('explicit_reserved_usage')
    # A later corpus version may join old and new records into a renamed
    # component. Propagate matches through the input's full group identities,
    # including bridges that are not themselves selected for evaluation.
    by_group, memberships = defaultdict(list), []
    for i, row in enumerate(rows):
        groups=set()
        for key in (*GROUP_KEYS, 'split_group_id'):
            for value in group_values(row,key):
                groups.add(('canonical_group' if key in {'split_group_id','corpus_split_group_id'} else key,value))
        memberships.append(groups)
        for group in groups:
            by_group[group].append(i)
    queue, visited = deque(reasons), set()
    while queue:
        i=queue.popleft()
        for group in memberships[i] - visited:
            visited.add(group)
            for j in by_group[group]:
                if j not in reasons:
                    reasons[j].add('registered_reservation_group_closure')
                    queue.append(j)
    return {i: sorted(values) for i, values in reasons.items()}


def assert_training_allowed(rows, *, allow_reserved_test=False):
    rows=list(rows)
    ledgers = registered_ledgers()
    blocked = protected_rows(rows, ledgers)
    if allow_reserved_test:
        blocked={i:reasons for i,reasons in blocked.items() if rows[i].get('split')!='test'}
    if blocked:
        raise ValueError(f'Reserved evaluation material cannot enter training/validation: {len(blocked)} rows')
    return {str(path): file_hash(path) for path, _ in ledgers}


def select_candidates(rows, count, seed, *, group_sizes=None, max_embargo=None):
    """Round-robin genre/source/length strata; hash order; prefer distinct groups.

    This is a diversity sample, not a probability sample of corpus prevalence.
    Unselected members of each chosen group remain embargoed.
    """
    buckets = defaultdict(list)
    for row in rows:
        length = row['quality']['words']
        band = next((str(limit) for limit in (40, 80, 160) if length < limit), '160+')
        key = (row.get('genre', ''), row.get('source_family') or row['source_domain'],
               row.get('upstream_domain') or '', band)
        buckets[key].append(row)
    queues = [deque(sorted(part, key=lambda r: digest(f'{seed}|{r["doc_id"]}')))
              for _, part in sorted(buckets.items())]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
    selected, seen, cost = [], set(), 0
    for row in ordered:
        group = row['split_group_id']
        size = group_sizes[group] if group_sizes is not None else 1
        if group not in seen and (max_embargo is None or cost + size <= max_embargo):
            selected.append(row); seen.add(group); cost += size
            if len(selected) == count:
                return selected
    selected_ids = {r['doc_id'] for r in selected}
    selected.extend(r for r in ordered if r['doc_id'] not in selected_ids
                    and r['split_group_id'] in seen)
    if len(selected) < count:
        raise ValueError(f'Insufficient reservation candidates: {len(selected)} < {count}')
    return selected[:count]


def allocate(rows, known, policy):
    """Close every reservation over the canonical full-document component."""
    old = {identity for row in known for identity in identities(row)}
    blocked = {r['split_group_id'] for r in rows if old.intersection(identities(r))}
    selectors = list(policy['source_selectors'].values())
    source_groups = {r['split_group_id'] for r in rows
                     if any(source_matches(r, selector) for selector in selectors)}
    novel = [r for r in rows if r['split_group_id'] not in blocked]
    chosen = []
    group_sizes = Counter(r['split_group_id'] for r in rows)
    for style in policy['styles']:
        source_pool = [r for r in novel if r['style'] == style
                       and source_matches(r, policy['source_selectors'][style])]
        documents = [r for r in novel if r['style'] == style
                     and r['split_group_id'] not in source_groups
                     and r.get('genre') not in policy.get('training_only_genres', [])]
        for role, pool in [('source_shift', source_pool), ('document_groups', documents)]:
            budget = policy.get('document_group_embargo_budget_per_style') if role == 'document_groups' else None
            for row in select_candidates(pool, policy['candidates_per_style_per_role'], policy['seed'],
                                         group_sizes=group_sizes, max_embargo=budget):
                chosen.append({'doc_id': row['doc_id'], 'style': style, 'role': role,
                               'split_group_id': row['split_group_id'],
                               'text_sha256': row['text_sha256']})
    chosen_groups = {r['split_group_id'] for r in chosen}
    forbidden = source_groups | chosen_groups
    protected = [r for r in rows if r['split_group_id'] in forbidden]
    eligible = [r for r in rows if r['split_group_id'] not in forbidden
                and not str(r.get('usage_status', '')).startswith('reserved')]
    for style in policy['styles']:
        if sum(r['style'] == style for r in eligible) < policy.get('min_remaining_per_style', 1):
            raise ValueError(f'Reservation leaves too few development documents: {style}')
    if {r['split_group_id'] for r in eligible} & forbidden:
        raise AssertionError('Reservation closure failed')
    return chosen, protected, eligible, blocked


def build(policy_path, out, ledger_path):
    from kazstyle.data.export_corpus import load_documents, verify_integrity
    policy_path, out, ledger_path = Path(policy_path), Path(out), Path(ledger_path)
    if out.exists() or ledger_path.exists():
        raise FileExistsError('Use new paths for a new reservation')
    source_sha = file_hash(Path(__file__))
    policy_sha = file_hash(policy_path)
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    corpus = project_path(policy['corpus'])
    if file_hash(corpus/'COMPLETE.json') != policy['corpus_complete_sha256']:
        raise ValueError('Reservation policy names a different corpus')
    guidelines = project_path(policy['guidelines'])
    if file_hash(guidelines) != policy['guidelines_sha256']:
        raise ValueError('Annotation guidelines differ from reservation policy')
    rows = load_documents(corpus)
    known = []
    for name, sha in policy['known_development_inputs'].items():
        path = project_path(name)
        if file_hash(path) != sha:
            raise ValueError('Known development input changed')
        known.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    chosen, protected, eligible, blocked = allocate(rows, known, policy)
    ledger = {'version': VERSION, 'reservation_id': policy['reservation_id'],
              'corpus': policy['corpus'], 'corpus_complete_sha256': policy['corpus_complete_sha256'],
              'policy_sha256': policy_sha, 'guidelines_sha256': policy['guidelines_sha256'],
              'status': 'reserved_pending_exact_view_and_label_review',
              'source_selectors': list(policy['source_selectors'].values()),
              'protected_identities': _index(protected), 'candidates': chosen,
              'expert_verified': False, 'model_predictions_used': False,
              'known_development_inputs': policy['known_development_inputs'],
              'limitations': policy['limitations']}
    guard = protected_rows(rows, [(None, ledger)])
    if {rows[i]['doc_id'] for i in guard} != {r['doc_id'] for r in protected}:
        raise ValueError('Reservation identities block a different set than canonical group closure')
    counts = lambda items: dict(sorted(Counter(r['style'] for r in items).items()))
    summary = {'reservation_id': policy['reservation_id'], 'corpus_documents': len(rows),
               'candidates': len(chosen), 'candidate_counts': {
                   role: counts([r for r in chosen if r['role'] == role])
                   for role in ('document_groups', 'source_shift')},
               'embargo_documents': len(protected), 'embargo_by_style': counts(protected),
               'eligible_development_documents': len(eligible), 'eligible_by_style': counts(eligible),
               'groups_touching_known_development': len(blocked),
               'evaluation_labels': 'provisional_pending_exact_view_review',
               'final_reference_ready': False, 'new_training_started': False}
    verify_integrity(corpus)
    if file_hash(policy_path) != policy_sha or file_hash(guidelines) != policy['guidelines_sha256']:
        raise ValueError('Reservation inputs changed during construction')
    for name, sha in policy['known_development_inputs'].items():
        if file_hash(project_path(name)) != sha:
            raise ValueError('Known development input changed during construction')
    if file_hash(Path(__file__)) != source_sha:
        raise ValueError('Reservation code changed during construction')
    out.mkdir(parents=True); ledger_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out/'code_snapshot.py')
    if file_hash(out/'code_snapshot.py') != source_sha:
        raise ValueError('Reservation code changed while taking its snapshot')
    shutil.copy2(policy_path, out/'policy.json'); shutil.copy2(guidelines, out/'guidelines.json')
    write_json(ledger_path, ledger)
    chosen_ids = {r['doc_id']: r['role'] for r in chosen}
    # Original texts are kept locally; the tracked compact ledger contains no prose.
    write_jsonl(out/'candidates.jsonl', [{**r, 'corpus_split_group_id': r['split_group_id'],
        'reservation_role': chosen_ids[r['doc_id']], 'usage_status': 'reserved_pending_review'}
        for r in rows if r['doc_id'] in chosen_ids])
    write_json(out/'summary.json', summary)
    write_json(out/'COMPLETE.json', {'ledger_sha256': file_hash(ledger_path),
        **{name: file_hash(out/name) for name in ('policy.json', 'guidelines.json',
                                                'candidates.jsonl', 'summary.json', 'code_snapshot.py')}})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.policy, args.out_dir, args.ledger)))
