"""Cached local inference for the testing UI. No network or input-text logging."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import joblib
import torch
from transformers import AutoConfig, AutoTokenizer

from kazstyle.data.corpus import file_hash, make_excerpt, normalize_text
from kazstyle.models.style_transformer import StyleTransformer
from kazstyle.settings import PROJECT_ROOT

ROOT = PROJECT_ROOT
MODELS = {
    'roberta': {'name': 'Kaz-RoBERTa', 'kind': 'transformer', 'run': 'pilot_v2_roberta_last'},
    'roberta_concat4': {'name': 'Kaz-RoBERTa · 4 слоя', 'kind': 'transformer', 'run': 'pilot_v2_roberta_concat4'},
    'logreg': {'name': 'TF-IDF + Logistic Regression', 'kind': 'classical', 'file': 'word_tfidf_logreg.joblib'},
    'word_svm': {'name': 'TF-IDF слов + SVM', 'kind': 'classical', 'file': 'word_tfidf_linear_svc.joblib'},
    'char_svm': {'name': 'TF-IDF символов + SVM', 'kind': 'classical', 'file': 'char_tfidf_linear_svc.joblib'},
}
RUSSIAN_STYLES = {'Formal': 'Официально-деловой', 'Publicist': 'Публицистический', 'Artistic': 'Художественный'}


class InferenceService:
    default_model = 'roberta'

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.dataset = self.root / 'data/processed/pilot_v2'
        self.config = json.loads((self.dataset / 'config.json').read_text(encoding='utf-8'))
        for name, expected in self.config['tokenizer_hashes'].items():
            if file_hash(self.dataset / 'tokenizer' / name) != expected:
                raise ValueError('Tokenizer differs from the training snapshot')
        self.tokenizer = AutoTokenizer.from_pretrained(self.dataset / 'tokenizer', local_files_only=True)
        self.cache = {}
        self.lock = threading.Lock()
        torch.set_num_threads(4)

    def available_models(self):
        return [{'id': key, 'name': spec['name']} for key, spec in MODELS.items()]

    def _load(self, key):
        if key in self.cache:
            return self.cache[key]
        spec = MODELS[key]
        if spec['kind'] == 'classical':
            bundle = joblib.load(self.root / 'artifacts/pilot_v2_classical' / spec['file'])
            if bundle['manifest_sha256'] != self.config['manifest_sha256']:
                raise ValueError('Model and dataset versions differ')
            model = bundle['pipeline']
        else:
            run = self.root / 'artifacts' / spec['run']
            checkpoint = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=True)
            if checkpoint['manifest_sha256'] != self.config['manifest_sha256']:
                raise ValueError('Model and dataset versions differ')
            encoder_config = AutoConfig.from_pretrained(run / 'encoder_config', local_files_only=True)
            model = StyleTransformer(checkpoint['model_name'], checkpoint['num_labels'], checkpoint['head'],
                                     encoder_config=encoder_config)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
        self.cache[key] = model
        return model

    def classify(self, text, model_key='roberta', compare=False):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Вставьте текст для проверки.')
        if len(text) > 100_000:
            raise ValueError('Текст слишком длинный. Максимум — 100 000 символов.')
        if model_key not in MODELS:
            raise ValueError('Выберите модель из списка.')
        if not any(c.isalpha() for c in text):
            raise ValueError('Нужен текст со словами, а не только числа или знаки.')
        cleaned = normalize_text(text)
        if not cleaned or not any(c.isalpha() for c in cleaned):
            raise ValueError('После удаления ссылок не осталось текста для проверки.')
        start = time.perf_counter()
        # Serialise model loading and inference to keep local CPU/memory usage bounded.
        with self.lock:
            excerpt, details = make_excerpt(cleaned, self.tokenizer,
                                            self.config['max_words'], self.config['max_tokens'])
            if not excerpt:
                raise ValueError('Не удалось выделить фрагмент. Попробуйте обычный связный текст.')
            keys = list(MODELS) if compare else [model_key]
            results = []
            for key in keys:
                tick = time.perf_counter()
                model = self._load(key)
                if MODELS[key]['kind'] == 'classical':
                    label = int(model.predict([excerpt])[0])
                else:
                    encoded = self.tokenizer(excerpt, return_tensors='pt', truncation=False)
                    if encoded['input_ids'].shape[1] > self.config['max_tokens']:
                        raise RuntimeError('Unexpected token limit mismatch')
                    with torch.inference_mode():
                        label = int(model(**encoded).argmax(-1).item())
                style = self.config['id_to_label'][str(label)]
                results.append({'model': key, 'model_name': MODELS[key]['name'], 'style': style,
                                'style_ru': RUSSIAN_STYLES[style],
                                'elapsed_ms': round((time.perf_counter() - tick) * 1000)})
        words = len(cleaned.split())
        warnings = []
        if words < 40:
            warnings.append('Короткий текст: модель обучалась на более длинных примерах. Ответ может быть неточным.')
        if len({r['style'] for r in results}) > 1:
            warnings.append('Модели не согласны между собой. Этот пример стоит разобрать отдельно.')
        return {'results': results, 'excerpt': excerpt, 'input_words': words,
                'analyzed_words': len(excerpt.split()), 'token_count': details['token_count'],
                'partial_text': excerpt != cleaned, 'warnings': warnings,
                'elapsed_ms': round((time.perf_counter() - start) * 1000)}
