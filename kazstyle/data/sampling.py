"""Source/genre sampling without tokenizers, model weights or ML runtimes."""
import pandas as pd


def diverse_sample(part, n, seed):
    """Round-robin strata; preserve missing metadata as a stratum, never duplicate."""
    if type(n) is not int or n < 0:
        raise ValueError('Sample size must be a non-negative integer')
    if n == 0 or part.empty:
        return part.iloc[:0].copy()
    buckets = [group.sample(frac=1, random_state=seed).to_dict('records')
               for _, group in part.groupby(['source_domain', 'genre'], sort=True, dropna=False)]
    chosen = []
    while buckets and len(chosen) < n:
        next_buckets = []
        for bucket in buckets:
            if len(chosen) < n:
                chosen.append(bucket.pop())
            if bucket:
                next_buckets.append(bucket)
        buckets = next_buckets
    return pd.DataFrame(chosen, columns=part.columns)
