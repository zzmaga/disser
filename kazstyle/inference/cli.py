"""Classify a pasted text or UTF-8 file using the website's local inference service."""
import argparse
import json
from pathlib import Path

from kazstyle.inference.service import InferenceService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text')
    inputs.add_argument('--text-file', type=Path)
    parser.add_argument('--model', default=None)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    text = args.text if args.text is not None else args.text_file.read_text(encoding='utf-8-sig')
    service = InferenceService()
    try:
        result = service.classify(text, args.model or service.default_model, args.compare)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
