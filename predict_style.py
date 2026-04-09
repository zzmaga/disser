import sys

from train_compare_styles import (
    DATA_FILES,
    MAX_CHARS,
    TfidfCentroidClassifier,
    balance_dataset,
    build_vocabulary,
    load_dataset,
    normalize_text,
    tokenize_char_ngrams,
    trim_text,
    vectorize_counts,
)


def train_demo_model():
    dataset = load_dataset()
    balanced = balance_dataset(dataset, seed=42)

    tokenized_docs = [
        tokenize_char_ngrams(row["text"]) for row in balanced.to_dict("records")
    ]
    vocabulary = build_vocabulary(tokenized_docs, max_features=25000)
    vectors = vectorize_counts(tokenized_docs, vocabulary)
    labels = balanced["label"].tolist()

    model = TfidfCentroidClassifier()
    model.fit(vectors, labels)
    return model, vocabulary


def predict_text(model, vocabulary, text):
    cleaned = trim_text(normalize_text(text), MAX_CHARS)
    tokens = tokenize_char_ngrams(cleaned)
    vector = vectorize_counts([tokens], vocabulary)[0]
    scores = model.score_one(vector)
    predicted_label = max(scores.items(), key=lambda item: item[1])[0]
    return cleaned, predicted_label, scores


def read_input_text():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    print("Paste text. Finish with an empty line:")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    print("Training demo classifier on current datasets...")
    print(f"Datasets: {', '.join(f'{k}={v}' for k, v in DATA_FILES.items())}")
    model, vocabulary = train_demo_model()

    text = read_input_text()
    if not text:
        raise SystemExit("No text provided.")

    cleaned, predicted_label, scores = predict_text(model, vocabulary, text)

    print("\nPrediction:")
    print(f"  predicted_style = {predicted_label}")
    print(f"  text_length = {len(cleaned)}")
    print("  scores:")
    for label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"    {label}: {value:.6f}")


if __name__ == "__main__":
    main()
