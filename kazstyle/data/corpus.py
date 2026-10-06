"""Strict ingestion and a balanced, document-disjoint style pilot.

Labels are source-derived and unreviewed. Dataset construction never claims
that source identity is expert style annotation.
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

STYLES = {"official": "Formal", "scientific": "Scientific", "publicistic": "Publicist",
          "literary": "Artistic", "colloquial": "Colloquial"}
GLOBAL_IDS = {name: i for i, name in enumerate(STYLES)}
ALIASES = {"formal": "official", "publicist": "publicistic", "artistic": "literary"}
DEFAULT_MODEL = "kz-transformers/kaz-roberta-conversational"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid source URL: {value!r}")
    # Keep path and query: distinct legal documents must not be merged by guesswork.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def read_csv_strict(path: Path):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="strict")
    csv.field_size_limit(2**31 - 1)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader)
    if len(set(header)) != len(header):
        raise ValueError(f"Duplicate columns in {path}")
    if not {"text", "label", "source_url"}.issubset(header):
        raise ValueError(f"Required columns absent in {path}: {header}")
    rows = []
    for number, values in enumerate(reader, 1):
        if len(values) != len(header):
            raise ValueError(f"{path}: record {number}, physical line {reader.line_num}: "
                             f"expected {len(header)} fields, received {len(values)}")
        row = dict(zip(header, values))
        row["source_row"] = number
        rows.append(row)
    words = np.asarray([len(row["text"].split()) for row in rows])
    audit = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
             "encoding": "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8",
             "delimiter": ",", "columns": header, "records": len(rows),
             "physical_lines": len(text.splitlines()),
             "embedded_newline_records": sum("\n" in r["text"] or "\r" in r["text"] for r in rows),
             "empty_fields": {key: sum(not r[key].strip() for r in rows) for key in header},
             "word_quantiles": dict(zip(["min", "p10", "median", "p90", "max"],
                                         np.quantile(words, [0, .1, .5, .9, 1]).tolist())) if len(words) else {},
             "exact_text_duplicates": len(rows) - len({r["text"] for r in rows})}
    return rows, audit


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    # Only actual HTML markup is removed; names such as Абай remain content.
    if re.search(r"</?(?:p|div|span|script|style|br|a)\b", text, re.I):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        text = soup.get_text(" ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_]{3,}", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def normalized_hash(text: str) -> str:
    return digest(re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip())


def language_signals(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    kaz = len(re.findall("[ӘәҒғҚқҢңӨөҰұҮүҺһІі]", text))
    cyr = len(re.findall("[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", text))
    return {"kazakh_specific_chars": kaz, "cyrillic_letter_fraction": cyr / max(1, len(letters)),
            "language_status": "script_screen_only_not_language_verification"}


def make_excerpt(text: str, tokenizer, max_words=160, max_tokens=256):
    """One central, non-overlapping sample per document, identical for all models."""
    matches = list(re.finditer(r"\S+", text))
    start_word = max(0, (len(matches) - max_words) // 2)
    end_word = min(len(matches), start_word + max_words)
    if not matches:
        return "", {"word_start": 0, "tokens_before_limit": 0, "token_count": 0}
    start_char = matches[start_word].start()
    excerpt = text[start_char:matches[end_word - 1].end()]
    before = len(tokenizer(excerpt, add_special_tokens=True)["input_ids"])
    while True:
        encoded = tokenizer(excerpt, add_special_tokens=True)
        if len(encoded["input_ids"]) <= max_tokens:
            break
        # Cut by tokenizer offsets, then keep a complete whitespace-delimited word.
        truncated = tokenizer(excerpt, add_special_tokens=True, truncation=True,
                              max_length=max_tokens, return_offsets_mapping=True)
        end = max(b for a, b in truncated["offset_mapping"])
        boundaries = [m.end() for m in re.finditer(r"\S+", excerpt) if m.end() <= end]
        if not boundaries:
            excerpt = ""
            break
        new_excerpt = excerpt[:boundaries[-1]].rstrip()
        if new_excerpt == excerpt:
            new_excerpt = " ".join(excerpt.split()[:-1])
        excerpt = new_excerpt
    return excerpt, {"word_start": start_word, "char_start": start_char,
                     "char_end": start_char + len(excerpt), "tokens_before_limit": before,
                     "token_count": len(tokenizer(excerpt, add_special_tokens=True)["input_ids"]),
                     "excerpt_words": len(excerpt.split())}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def duplicate_groups(frame: pd.DataFrame, threshold=.92):
    """Conservative candidate groups; not a guarantee of eliminating all duplicates.

    Stateless char hashing avoids learning a feature vocabulary from held-out data.
    Connected components keep near-duplicate families within a single split.
    """
    uf = UnionFind(len(frame))
    edges = []
    for key in ["doc_id", "content_hash", "excerpt_hash"]:
        first = {}
        for i, value in enumerate(frame[key]):
            if value in first:
                uf.union(i, first[value])
                edges.append({"a": i, "b": first[value], "kind": key, "similarity": 1.0})
            else:
                first[value] = i
    for field in ["text", "duplicate_probe"]:
        probe = frame[field].map(lambda s: re.sub(r"\d+", "0", s.casefold()))
        vectorizer = HashingVectorizer(analyzer="char", ngram_range=(5, 5),
                                      n_features=2**18, alternate_sign=False, norm="l2", dtype=np.float32)
        matrix = vectorizer.transform(probe)
        for start in range(0, len(frame), 128):
            similarity = (matrix[start:start + 128] @ matrix.T).tocoo()
            keep = (similarity.data >= threshold) & (similarity.col > similarity.row + start)
            for local, j, score in zip(similarity.row[keep], similarity.col[keep], similarity.data[keep]):
                i = int(local) + start
                uf.union(i, int(j))
                edges.append({"a": i, "b": int(j), "kind": field, "similarity": float(score)})
    members = {}
    for i in range(len(frame)):
        members.setdefault(uf.find(i), []).append(frame.iloc[i]["doc_id"])
    names = {root: digest("|".join(sorted(set(ids))))[:24] for root, ids in members.items()}
    return [names[uf.find(i)] for i in range(len(frame))], edges


def balanced_splits(frame, seed=42, per_class=300, group_key='duplicate_group_id'):
    """Stratify independent groups, then balance document counts within each split.

    Surplus records stay in reserve; no oversampling and no cross-split group reuse.
    """
    groups = frame.groupby(group_key).agg(label=("label", "first"), n_labels=("label", "nunique"))
    if (groups.n_labels != 1).any():
        raise ValueError(f"Mixed-label {group_key} cannot be split with single-label stratification; review grouping or use a separately specified multilabel allocation")
    train_groups, rest = train_test_split(groups.index.to_numpy(), test_size=.30,
                                         random_state=seed, stratify=groups.label)
    val_groups, test_groups = train_test_split(rest, test_size=.5, random_state=seed,
                                              stratify=groups.loc[rest, "label"])
    split_sets = dict(train=set(train_groups), validation=set(val_groups), test=set(test_groups))
    parts, counts = [], {}
    targets = {"train": int(per_class * .7), "validation": int(per_class * .15)}
    targets["test"] = per_class - sum(targets.values())
    labels = sorted(frame.label.unique())
    for split, ids in split_sets.items():
        part = frame[frame[group_key].isin(ids)]
        present = part.label.value_counts()
        if set(present.index) != set(labels):
            raise ValueError(f"Missing class in {split}")
        count = min(targets[split], int(present.min()))
        if count < 5:
            raise ValueError(f"Too few independent examples for {split}: {count}")
        selected = pd.concat([part[part.label == label].sample(count, random_state=seed)
                              for label in labels]).copy()
        selected["split"] = split
        parts.append(selected)
        counts[split] = {"documents_per_class": count, "groups": selected[group_key].nunique()}
    result = pd.concat(parts).sort_values(["split", "label", "doc_id"]).reset_index(drop=True)
    validate_manifest(result)
    return result, counts


def validate_manifest(frame):
    required = {"sample_id", "doc_id", "duplicate_group_id", "excerpt_hash", "text", "label", "split"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Missing manifest fields/values")
    if frame.sample_id.duplicated().any() or frame.doc_id.duplicated().any():
        raise ValueError("Pilot expects exactly one sample per document")
    for key in ["doc_id", "duplicate_group_id", "excerpt_hash"]:
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Cross-split leakage: {key}")
    if 'split_group_id' in frame:
        from kazstyle.data.grouping import validate_group_separation
        validate_group_separation(frame)
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test")
    labels = set(frame.label)
    for _, part in frame.groupby("split"):
        if set(part.label) != labels:
            raise ValueError("Class absent in split")


def build_corpus(data_dir: Path, out: Path, styles: list[str], per_class=300, seed=42,
                 max_words=160, max_tokens=256, min_words=40, near_threshold=.92,
                 model_name=DEFAULT_MODEL, local_only=True):
    from transformers import AutoTokenizer
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite dataset version: {out}")
    if len(styles) not in {2, 3} or len(set(styles)) != len(styles) or any(s not in STYLES for s in styles):
        raise ValueError("Choose 2 or 3 distinct known styles for the pilot")
    if per_class < 40 or max_tokens < 32 or max_words < min_words or not .5 <= near_threshold <= 1:
        raise ValueError("Invalid size/length/similarity parameters")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    if not tokenizer.is_fast:
        raise ValueError("An offset-capable fast tokenizer is required")
    out.mkdir(parents=True)
    tokenizer.save_pretrained(out / "tokenizer")
    mapping = {style: i for i, style in enumerate(styles)}
    audit, records, excluded = {}, [], []
    for style in STYLES:
        path = data_dir / f"{style}.csv"
        rows, audit[style] = read_csv_strict(path)
        print(f"[read] {style}: {len(rows)} records", flush=True)
        for row in rows:
            label = ALIASES.get(row["label"].strip().lower(), row["label"].strip().lower())
            if label != style:
                raise ValueError(f"File/label mismatch: {path}, row {row['source_row']}")
            if style not in styles:
                continue
            url = canonical_url(row["source_url"])
            doc_id = digest(url)[:24]
            cleaned = normalize_text(row["text"])
            wc = len(cleaned.split())
            provenance = {"doc_id": doc_id, "source_url": url, "source_file": str(path),
                          "source_row": row["source_row"], "style": style}
            if wc < min_words or wc > 100_000:
                excluded.append({**provenance, "reason": "document_length_review", "words": wc})
                continue
            signals = language_signals(cleaned)
            if signals["cyrillic_letter_fraction"] < .6 or signals["kazakh_specific_chars"] < 2:
                excluded.append({**provenance, "reason": "language_review", **signals})
                continue
            excerpt, length = make_excerpt(cleaned, tokenizer, max_words, max_tokens)
            if len(excerpt.split()) < 20:
                excluded.append({**provenance, "reason": "excerpt_too_short", **length})
                continue
            # Head+middle+tail signature supplements matching the model input window.
            words = cleaned.split()
            probe = " ".join(words[:120] + words[max(0, len(words)//2-60):len(words)//2+60] + words[-120:])
            records.append({**provenance, "sample_id": digest(doc_id + "|" + excerpt)[:24],
                            "label": mapping[style], "global_label": GLOBAL_IDS[style],
                            "style_name": STYLES[style], "text": excerpt, "text_clean": cleaned,
                            "text_raw": row["text"], "source_domain": urlsplit(url).netloc,
                            "site": row.get("site", "").strip() or urlsplit(url).netloc,
                            "year": row.get("year", "") or None, "collected_at": None,
                            "label_origin": "source_heuristic", "review_status": "unreviewed",
                            "raw_words": len(row["text"].split()), "clean_words": wc,
                            "content_hash": normalized_hash(cleaned), "excerpt_hash": normalized_hash(excerpt),
                            "duplicate_probe": probe, **length, **signals})
    frame = pd.DataFrame(records).sort_values(["style", "doc_id"]).reset_index(drop=True)
    print(f"[dedup] grouping {len(frame)} documents", flush=True)
    frame["duplicate_group_id"], edges = duplicate_groups(frame, near_threshold)
    conflict_groups = set(frame.groupby("duplicate_group_id").label.nunique().loc[lambda s: s > 1].index)
    conflicting = frame[frame.duplicate_group_id.isin(conflict_groups)]
    for row in conflicting.to_dict("records"):
        excluded.append({"doc_id": row["doc_id"], "reason": "duplicate_label_conflict",
                         "duplicate_group_id": row["duplicate_group_id"], "style": row["style"]})
    candidates = frame[~frame.duplicate_group_id.isin(conflict_groups)].copy()
    candidates = candidates.drop_duplicates("doc_id").drop_duplicates("content_hash").drop_duplicates("excerpt_hash")
    selected, counts = balanced_splits(candidates, seed, per_class)
    excluded_ids = set(selected.doc_id)
    reserve = candidates[~candidates.doc_id.isin(excluded_ids)]
    # Preserve full documents only in the provenance store; models read manifest.text.
    write_jsonl(out / "documents.jsonl", frame.drop(columns="duplicate_probe").to_dict("records"))
    manifest = selected.drop(columns=["text_raw", "text_clean", "duplicate_probe"])
    write_jsonl(out / "manifest.jsonl", manifest.to_dict("records"))
    manifest.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    write_jsonl(out / "quarantine.jsonl", excluded)
    reserve[["doc_id", "style", "duplicate_group_id"]].to_csv(out / "reserve.csv", index=False)
    write_json(out / "duplicate_edges.json", [dict(a=frame.iloc[e['a']].doc_id,
               b=frame.iloc[e['b']].doc_id, kind=e['kind'], similarity=e['similarity']) for e in edges])
    review = []
    for _, part in selected[selected.split == "train"].groupby("label"):
        for row in part.sample(min(20, len(part)), random_state=seed).to_dict("records"):
            review.append({k: row[k] for k in ["doc_id", "source_url", "style", "text", "text_clean"]} |
                          {"manual_label": None, "reviewer": None, "comment": None})
    write_jsonl(out / "annotation_sample_train.jsonl", review)
    config = {"schema_version": 2, "styles": styles, "label_to_id": mapping,
              "id_to_label": {str(i): STYLES[s] for s, i in mapping.items()},
              "seed": seed, "requested_per_class": per_class, "max_words": max_words,
              "max_tokens": max_tokens, "min_document_words": min_words, "near_duplicate_threshold": near_threshold,
              "excerpt_policy": "one central token-limited excerpt per document",
              "tokenizer_name": model_name, "tokenizer_hashes": {p.name:file_hash(p) for p in (out/'tokenizer').iterdir() if p.is_file()},
              "manifest_sha256": file_hash(out / "manifest.jsonl"), "split_counts": counts,
              "limitations": ["Labels are source-derived, not manually verified.",
                              "Each pilot class originates from one website; source/style confounding remains.",
                              "Near-duplicate detection is a candidate heuristic, not an exhaustive guarantee.",
                              "One central excerpt does not represent all possible passages of a long document.",
                              "Kazakh language is screened by script only, not expert verification."]}
    write_json(out / "config.json", config)
    write_json(out / "raw_audit.json", audit)
    summary = {"eligible_unique_documents": len(candidates), "selected_documents": len(selected),
               "quarantined_records": len(excluded), "reserve_documents": len(reserve),
               "near_or_exact_edges": len(edges), "selected_groups": selected.duplicate_group_id.nunique(),
               "excerpt_token_quantiles": selected.token_count.quantile([0,.5,.9,1]).to_dict(),
               "token_limited_examples": int((selected.tokens_before_limit > max_tokens).sum()),
               "source_by_style": pd.crosstab(selected.source_domain, selected['style']).to_dict()}
    write_json(out / "summary.json", summary)
    write_json(out / "COMPLETE.json", {"manifest_sha256": config['manifest_sha256']})
    return config, summary


def load_manifest(dataset: Path):
    if not (dataset / "COMPLETE.json").exists():
        raise ValueError("Dataset build is incomplete; refuse training")
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    if file_hash(dataset / "manifest.jsonl") != config["manifest_sha256"]:
        raise ValueError("Manifest changed after dataset construction")
    frame = pd.read_json(dataset / "manifest.jsonl", lines=True, dtype={"sample_id":str,"doc_id":str,"excerpt_hash":str,"duplicate_group_id":str})
    validate_manifest(frame)
    if config.get('schema_version') == 3:
        from kazstyle.data.quality import assert_text_only
        if file_hash(dataset/'metadata.jsonl') != config['metadata_sha256']:
            raise ValueError('Provenance metadata changed after dataset construction')
        for split in ['train','validation','test']:
            path=dataset/f'{split}.csv'
            if file_hash(path)!=config['input_file_hashes'][path.name]:
                raise ValueError('Text-only input file changed after dataset construction')
            inputs=pd.read_csv(path,encoding='utf-8',keep_default_na=False)
            if list(inputs.columns)!=['text','label']:
                raise ValueError('Only text and label may appear in training input files')
            expected=frame[frame.split==split]
            if inputs.text.tolist()!=expected.text.tolist() or inputs.label.tolist()!=expected.label.tolist():
                raise ValueError('Model inputs and report manifest differ')
            assert_text_only(inputs.text)
    return frame, config
