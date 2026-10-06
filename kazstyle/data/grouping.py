"""Keep known document, work, author and template families in one split.

Split groups are distinct from duplicate groups: two stories by one author are
not duplicate texts. Unknown authors never form a shared 'unknown' component.
"""
from collections import Counter, defaultdict
import pandas as pd
from kazstyle.data.corpus import UnionFind, digest
from kazstyle.data.overlap import words, shingles, overlap_metrics

GROUP_KEYS = ('duplicate_group_id', 'parent_group', 'author_group', 'work_group', 'template_family_id', 'passage_group_id')


def assign_passage_groups(frame):
    """Conservatively co-locate shared passages, without calling them duplicates.

    Uses the predeclared overlap audit thresholds on the exact model inputs.
    It never uses labels, predictions or the eventual split assignment.
    """
    uf = UnionFind(len(frame)); index = defaultdict(list); exact = {}; sizes = []; edges = []
    ids = frame.doc_id.tolist()
    for i, text in enumerate(frame.text):
        tokens = words(text); grams = shingles(tokens); normalized = digest(' '.join(tokens))
        hits = Counter(j for gram in grams for j in index.get(gram, ()))
        for j, shared in hits.items():
            metrics = overlap_metrics(shared, sizes[j], len(grams))
            if metrics:
                uf.union(i,j)
                edges.append({'left_doc_id':ids[j], 'right_doc_id':ids[i], **metrics})
        if normalized in exact:
            j = exact[normalized]; uf.union(i,j)
            if j not in hits:
                edges.append({'left_doc_id':ids[j], 'right_doc_id':ids[i], 'flag':'normalized_exact'})
        else:
            exact[normalized] = i
        sizes.append(len(grams))
        for gram in grams:
            index[gram].append(i)
    members = defaultdict(list)
    for i, doc_id in enumerate(ids):
        members[uf.find(i)].append(doc_id)
    names = {root:digest('passage|'+'|'.join(sorted(items)))[:24] if len(items)>1 else None
             for root,items in members.items()}
    return [names[uf.find(i)] for i in range(len(ids))], edges


def known_group(value):
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str):
        raise ValueError('Grouping identifiers must be strings or null')
    return value.strip() or None


def assign_split_groups(frame):
    uf = UnionFind(len(frame))
    for key in GROUP_KEYS:
        if key not in frame:
            continue
        first = {}
        for i, value in enumerate(frame[key]):
            value = known_group(value)
            if value is None:
                continue
            if value in first:
                uf.union(i, first[value])
            else:
                first[value] = i
    members = {}
    for i, doc_id in enumerate(frame['doc_id']):
        members.setdefault(uf.find(i), []).append(str(doc_id))
    names = {root: digest('split_family|' + '|'.join(sorted(set(ids))))[:24] for root, ids in members.items()}
    return [names[uf.find(i)] for i in range(len(frame))]


def validate_group_separation(frame):
    for key in (*GROUP_KEYS, 'split_group_id'):
        if key not in frame:
            continue
        values = frame[key].map(known_group)
        present = frame.loc[values.notna(), ['split']].assign(group=values[values.notna()])
        if (present.groupby('group')['split'].nunique() > 1).any():
            raise ValueError(f'Cross-split overlap: {key}')
