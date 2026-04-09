import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

DATA_FILES = {
    "official": "data/official.csv",
    "publicistic": "data/publicistic.csv",
}

OUTPUT_DIR = "results"
RANDOM_SEED = 42
MAX_CHARS = 3000
TRAIN_RATIO = 0.8


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def trim_text(text, max_chars):
    return text[:max_chars].strip()


def tokenize_words(text):
    return re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)


def tokenize_char_ngrams(text, min_n=3, max_n=5):
    compact = f" {text} "
    features = []
    for n in range(min_n, max_n + 1):
        for i in range(len(compact) - n + 1):
            features.append(compact[i : i + n])
    return features


def load_dataset():
    rows = []
    for label, path in DATA_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset: {path}")

        df = pd.read_csv(path)
        for text in df["text"].dropna():
            cleaned = trim_text(normalize_text(text), MAX_CHARS)
            if len(cleaned) >= 200:
                rows.append({"text": cleaned, "label": label})

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    return dataset


def balance_dataset(df, seed):
    rng = random.Random(seed)
    grouped = []
    min_size = df["label"].value_counts().min()

    for label in sorted(df["label"].unique()):
        items = df[df["label"] == label].to_dict("records")
        rng.shuffle(items)
        grouped.extend(items[:min_size])

    balanced = pd.DataFrame(grouped)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def stratified_split(df, train_ratio, seed):
    rng = random.Random(seed)
    train_rows = []
    test_rows = []

    for label in sorted(df["label"].unique()):
        items = df[df["label"] == label].to_dict("records")
        rng.shuffle(items)
        split_index = max(1, int(len(items) * train_ratio))
        train_rows.extend(items[:split_index])
        test_rows.extend(items[split_index:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def build_vocabulary(tokenized_docs, max_features):
    frequencies = Counter()
    for tokens in tokenized_docs:
        frequencies.update(tokens)
    return {
        token
        for token, _ in frequencies.most_common(max_features)
        if token and not token.isspace()
    }


def vectorize_counts(tokenized_docs, vocabulary):
    vectors = []
    for tokens in tokenized_docs:
        counts = Counter(token for token in tokens if token in vocabulary)
        vectors.append(counts)
    return vectors


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.labels = []
        self.class_doc_counts = Counter()
        self.class_token_counts = defaultdict(Counter)
        self.class_total_tokens = Counter()
        self.vocabulary = set()

    def fit(self, vectors, labels):
        self.labels = sorted(set(labels))
        self.class_doc_counts.update(labels)

        for vector, label in zip(vectors, labels):
            self.class_token_counts[label].update(vector)
            self.class_total_tokens[label] += sum(vector.values())

        for vector in vectors:
            self.vocabulary.update(vector.keys())

    def predict_one(self, vector):
        total_docs = sum(self.class_doc_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        best_label = None
        best_score = -float("inf")

        for label in self.labels:
            score = math.log(self.class_doc_counts[label] / total_docs)
            denom = self.class_total_tokens[label] + self.alpha * vocab_size

            for token, count in vector.items():
                numerator = self.class_token_counts[label][token] + self.alpha
                score += count * math.log(numerator / denom)

            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict(self, vectors):
        return [self.predict_one(vector) for vector in vectors]

    def top_features(self, limit=15):
        if len(self.labels) != 2:
            return []

        left, right = self.labels
        vocab = sorted(self.vocabulary)
        vocab_size = max(1, len(vocab))
        results = []

        for token in vocab:
            left_score = math.log(
                (self.class_token_counts[left][token] + self.alpha)
                / (self.class_total_tokens[left] + self.alpha * vocab_size)
            )
            right_score = math.log(
                (self.class_token_counts[right][token] + self.alpha)
                / (self.class_total_tokens[right] + self.alpha * vocab_size)
            )
            results.append((token, left_score - right_score))

        top_left = [token for token, _ in sorted(results, key=lambda x: x[1], reverse=True)[:limit]]
        top_right = [token for token, _ in sorted(results, key=lambda x: x[1])[:limit]]
        return [
            {"label": left, "features": top_left},
            {"label": right, "features": top_right},
        ]


class TfidfCentroidClassifier:
    def __init__(self):
        self.labels = []
        self.idf = {}
        self.centroids = {}

    def fit(self, vectors, labels):
        self.labels = sorted(set(labels))
        n_docs = len(vectors)
        document_frequency = Counter()

        for vector in vectors:
            document_frequency.update(vector.keys())

        self.idf = {
            token: math.log((1 + n_docs) / (1 + df)) + 1.0
            for token, df in document_frequency.items()
        }

        class_vectors = defaultdict(list)
        for vector, label in zip(vectors, labels):
            tfidf_vector = self._tfidf(vector)
            class_vectors[label].append(tfidf_vector)

        self.centroids = {}
        for label, items in class_vectors.items():
            centroid = Counter()
            for item in items:
                centroid.update(item)
            scale = max(1, len(items))
            for token in list(centroid.keys()):
                centroid[token] /= scale
            self.centroids[label] = self._normalize(centroid)

    def _tfidf(self, counts):
        tfidf = Counter()
        total = max(1, sum(counts.values()))
        for token, count in counts.items():
            tf = count / total
            tfidf[token] = tf * self.idf.get(token, 0.0)
        return self._normalize(tfidf)

    def _normalize(self, vector):
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return vector
        return Counter({token: value / norm for token, value in vector.items()})

    def _cosine(self, left, right):
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    def predict_one(self, vector):
        scores = self.score_one(vector)
        return max(scores.items(), key=lambda item: item[1])[0]

    def score_one(self, vector):
        tfidf_vector = self._tfidf(vector)
        return {
            label: self._cosine(tfidf_vector, centroid)
            for label, centroid in self.centroids.items()
        }

    def predict(self, vectors):
        return [self.predict_one(vector) for vector in vectors]


def compute_metrics(y_true, y_pred, labels):
    confusion = {label: {inner: 0 for inner in labels} for label in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[true_label][pred_label] += 1

    correct = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label)
    accuracy = correct / max(1, len(y_true))

    per_label = {}
    f1_values = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }
        f1_values.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 4),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def prepare_feature_sets(train_rows, test_rows, tokenizer, max_features):
    train_tokens = [tokenizer(row["text"]) for row in train_rows]
    test_tokens = [tokenizer(row["text"]) for row in test_rows]
    vocabulary = build_vocabulary(train_tokens, max_features=max_features)
    train_vectors = vectorize_counts(train_tokens, vocabulary)
    test_vectors = vectorize_counts(test_tokens, vocabulary)
    return train_vectors, test_vectors


def run_experiment(train_rows, test_rows):
    labels = sorted({row["label"] for row in train_rows})
    y_train = [row["label"] for row in train_rows]
    y_test = [row["label"] for row in test_rows]

    experiments = [
        {
            "name": "word_multinomial_nb",
            "tokenizer": tokenize_words,
            "max_features": 15000,
            "model": MultinomialNB(alpha=1.0),
        },
        {
            "name": "char_multinomial_nb",
            "tokenizer": tokenize_char_ngrams,
            "max_features": 25000,
            "model": MultinomialNB(alpha=1.0),
        },
        {
            "name": "word_tfidf_centroid",
            "tokenizer": tokenize_words,
            "max_features": 15000,
            "model": TfidfCentroidClassifier(),
        },
        {
            "name": "char_tfidf_centroid",
            "tokenizer": tokenize_char_ngrams,
            "max_features": 25000,
            "model": TfidfCentroidClassifier(),
        },
    ]

    results = []
    for experiment in experiments:
        train_vectors, test_vectors = prepare_feature_sets(
            train_rows,
            test_rows,
            tokenizer=experiment["tokenizer"],
            max_features=experiment["max_features"],
        )

        model = experiment["model"]
        model.fit(train_vectors, y_train)
        predictions = model.predict(test_vectors)
        metrics = compute_metrics(y_test, predictions, labels)

        result = {
            "model": experiment["name"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_label": metrics["per_label"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

        if hasattr(model, "top_features"):
            result["top_features"] = model.top_features(limit=12)

        results.append(result)

    results.sort(key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)
    return results


def save_results(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "style_comparison_results.json")
    md_path = os.path.join(OUTPUT_DIR, "style_comparison_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# Style Identification Report",
        "",
        f"- Date: {summary['created_at']}",
        f"- Labels: {', '.join(summary['labels'])}",
        f"- Balanced samples per label: {summary['balanced_samples_per_label']}",
        f"- Train size: {summary['train_size']}",
        f"- Test size: {summary['test_size']}",
        f"- Max chars per text: {summary['max_chars']}",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Macro F1 |",
        "|---|---:|---:|",
    ]

    for result in summary["results"]:
        lines.append(
            f"| {result['model']} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} |"
        )

    best = summary["results"][0]
    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"- Model: {best['model']}",
            f"- Accuracy: {best['accuracy']:.4f}",
            f"- Macro F1: {best['macro_f1']:.4f}",
            "",
            "## Confusion Matrix",
            "",
            "```json",
            json.dumps(best["confusion_matrix"], ensure_ascii=False, indent=2),
            "```",
        ]
    )

    if best.get("top_features"):
        lines.extend(["", "## Top Features", ""])
        for item in best["top_features"]:
            lines.append(f"- {item['label']}: {', '.join(item['features'])}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main():
    df = load_dataset()
    balanced = balance_dataset(df, seed=RANDOM_SEED)
    train_rows, test_rows = stratified_split(
        balanced,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED,
    )

    results = run_experiment(train_rows, test_rows)

    counts = balanced["label"].value_counts().to_dict()
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels": sorted(counts.keys()),
        "balanced_samples_per_label": min(counts.values()),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "max_chars": MAX_CHARS,
        "results": results,
    }

    json_path, md_path = save_results(summary)

    print("Dataset summary:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    print("\nModel comparison:")
    for result in results:
        print(
            f"  {result['model']}: accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            MAX_CHARS = int(sys.argv[1])
        except ValueError as exc:
            raise SystemExit("First argument must be an integer max_chars value.") from exc
    main()
