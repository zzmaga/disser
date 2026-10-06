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
from kazstyle.models.tokenization import tokenizer_specs,load_dataset_tokenizer,fit_shared_text
from kazstyle.settings import PROJECT_ROOT, artifact_path

ROOT = PROJECT_ROOT
MODELS = {
    'roberta': {'name': 'Kaz-RoBERTa', 'kind': 'transformer', 'run': 'pilot_v2_roberta_last'},
    'roberta_concat4': {'name': 'Kaz-RoBERTa · 4 слоя', 'kind': 'transformer', 'run': 'pilot_v2_roberta_concat4'},
    'logreg': {'name': 'TF-IDF + Logistic Regression', 'kind': 'classical', 'file': 'word_tfidf_logreg.joblib'},
    'word_svm': {'name': 'TF-IDF слов + SVM', 'kind': 'classical', 'file': 'word_tfidf_linear_svc.joblib'},
    'char_svm': {'name': 'TF-IDF символов + SVM', 'kind': 'classical', 'file': 'char_tfidf_linear_svc.joblib'},
}
RUSSIAN_STYLES = {'Formal': 'Официально-деловой', 'Publicist': 'Публицистический', 'Artistic': 'Художественный',
                  'Scientific':'Научный','Colloquial':'Разговорный'}


class InferenceService:
    default_model = 'roberta'

    def __init__(self, root=ROOT, deployment=None):
        self.root = Path(root)
        active=self.root/'configs/deployment.json'
        if deployment is None and active.exists():
            deployment=json.loads(active.read_text(encoding='utf-8'))
        elif isinstance(deployment,(str,Path)):
            deployment=json.loads(Path(deployment).read_text(encoding='utf-8'))
        deployment=deployment or {'dataset':'data/processed/pilot_v2','models':MODELS,'default_model':'roberta'}
        self.models=deployment['models']
        self.default_model=deployment['default_model']
        if self.default_model not in self.models:raise ValueError('Default model missing from deployment')
        self.dataset = self.root / deployment['dataset']
        self.config = json.loads((self.dataset / 'config.json').read_text(encoding='utf-8'))
        for name, expected in self.config['tokenizer_hashes'].items():
            if file_hash(self.dataset / 'tokenizer' / name) != expected:
                raise ValueError('Tokenizer differs from the training snapshot')
        self.tokenizer = AutoTokenizer.from_pretrained(self.dataset / 'tokenizer', local_files_only=True)
        self.constraints=[(self.tokenizer,self.config['max_tokens'])]
        for spec in tokenizer_specs(self.config)[1:]:
            tokenizer,_=load_dataset_tokenizer(self.dataset,self.config,spec['model_name'])
            self.constraints.append((tokenizer,spec['max_tokens']))
        self.run_tokenizers={}
        self.cache = {}
        self.lock = threading.Lock()
        torch.set_num_threads(4)

    def available_models(self):
        return [{'id': key, 'name': spec['name']} for key, spec in self.models.items()]

    def available_styles(self):
        return [RUSSIAN_STYLES.get(v,v) for v in self.config['id_to_label'].values()]

    def _load(self, key):
        if key in self.cache:
            return self.cache[key]
        spec = self.models[key]
        run = artifact_path(spec.get('run','pilot_v2_classical'), self.root)
        if spec.get('weights_sha256'):
            filename=spec['file'] if spec['kind']=='classical' else 'best_model.pt'
            if file_hash(run/filename)!=spec['weights_sha256']:
                raise ValueError('Model weights changed after deployment verification')
        if spec['kind'] == 'classical':
            bundle = joblib.load(run / spec['file'])
            if bundle['manifest_sha256'] != self.config['manifest_sha256']:
                raise ValueError('Model and dataset versions differ')
            model = bundle['pipeline']
        else:
            checkpoint = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=True)
            if checkpoint['manifest_sha256'] != self.config['manifest_sha256']:
                raise ValueError('Model and dataset versions differ')
            if 'tokenizer_hashes' in checkpoint:
                for name,sha in checkpoint['tokenizer_hashes'].items():
                    if file_hash(run/'tokenizer'/name)!=sha:raise ValueError('Run tokenizer changed after training')
                tokenizer=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True)
                self.run_tokenizers[key]=(tokenizer,checkpoint['max_tokens'])
            else:
                if checkpoint['model_name']!=self.config['tokenizer_name']:
                    raise ValueError('Legacy checkpoint has no matching verified tokenizer')
                self.run_tokenizers[key]=(self.tokenizer,self.config['max_tokens'])
            encoder_config = AutoConfig.from_pretrained(run / 'encoder_config', local_files_only=True)
            model = StyleTransformer(checkpoint['model_name'], checkpoint['num_labels'], checkpoint['head'],
                                     encoder_config=encoder_config)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
        self.cache[key] = model
        return model

    def classify(self, text, model_key=None, compare=False):
        if model_key is not None and not isinstance(model_key,str):
            raise ValueError('Некорректное название модели.')
        model_key=model_key or self.default_model
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Вставьте текст для проверки.')
        if len(text) > 100_000:
            raise ValueError('Текст слишком длинный. Максимум — 100 000 символов.')
        if model_key not in self.models:
            raise ValueError('Выберите модель из списка.')
        if not any(c.isalpha() for c in text):
            raise ValueError('Нужен текст со словами, а не только числа или знаки.')
        if self.config.get('cleaning_version') in {'text_only_v3','text_only_v4'}:
            from kazstyle.data.quality import clean_text, assert_text_only
            cleaned,_=clean_text(text)
            if cleaned:assert_text_only([cleaned])
        else:
            cleaned = normalize_text(text)
        if not cleaned or not any(c.isalpha() for c in cleaned):
            raise ValueError('После удаления ссылок не осталось текста для проверки.')
        start = time.perf_counter()
        # Serialise model loading and inference to keep local CPU/memory usage bounded.
        with self.lock:
            excerpt, details = make_excerpt(cleaned, self.tokenizer,
                                            self.config['max_words'], self.config['max_tokens'])
            if excerpt:
                excerpt=fit_shared_text(excerpt,self.constraints)
                details['token_count']=len(self.tokenizer(excerpt)['input_ids'])
            if not excerpt:
                raise ValueError('Не удалось выделить фрагмент. Попробуйте обычный связный текст.')
            keys = list(self.models) if compare else [model_key]
            results = []
            for key in keys:
                tick = time.perf_counter()
                model = self._load(key)
                if self.models[key]['kind'] == 'classical':
                    label = int(model.predict([excerpt])[0])
                else:
                    tokenizer,max_tokens=self.run_tokenizers[key]
                    encoded = tokenizer(excerpt, return_tensors='pt', truncation=False)
                    if encoded['input_ids'].shape[1] > max_tokens:
                        raise RuntimeError('Unexpected token limit mismatch')
                    with torch.inference_mode():
                        label = int(model(**encoded).argmax(-1).item())
                style = self.config['id_to_label'][str(label)]
                results.append({'model': key, 'model_name': self.models[key]['name'], 'style': style,
                                'style_ru': RUSSIAN_STYLES[style],
                                'elapsed_ms': round((time.perf_counter() - tick) * 1000)})
        words = len(cleaned.split())
        warnings = []
        if words < self.config.get('min_document_words',40):
            warnings.append('Короткий текст: модель обучалась на более длинных примерах. Ответ может быть неточным.')
        if len({r['style'] for r in results}) > 1:
            warnings.append('Модели не согласны между собой. Этот пример стоит разобрать отдельно.')
        return {'results': results, 'excerpt': excerpt, 'input_words': words,
                'analyzed_words': len(excerpt.split()), 'token_count': details['token_count'],
                'partial_text': excerpt != cleaned, 'warnings': warnings,
                'dataset_version':self.dataset.name,
                'elapsed_ms': round((time.perf_counter() - start) * 1000)}
