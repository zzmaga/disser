"""Build a NEW balanced two/three-style pilot from immutable raw CSV files."""
import argparse
from pathlib import Path
from kazstyle.settings import project_path
from kazstyle.data.corpus import build_corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=project_path('data'))
    p.add_argument('--out-dir', type=Path, default=project_path('data/processed/pilot_v2'))
    p.add_argument('--styles', nargs='+', default=['official','publicistic','literary'])
    p.add_argument('--per-class', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-words', type=int, default=160)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--min-words', type=int, default=40)
    p.add_argument('--near-threshold', type=float, default=.92)
    p.add_argument('--allow-download', action='store_true')
    args = p.parse_args()
    config, summary = build_corpus(args.data_dir, args.out_dir, args.styles, args.per_class,
                                  args.seed, args.max_words, args.max_tokens, args.min_words,
                                  args.near_threshold, local_only=not args.allow_download)
    print(config['split_counts'])
    print(summary)


if __name__ == '__main__':
    main()
