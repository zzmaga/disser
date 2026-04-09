# Style Identification Report

- Date: 2026-04-09T16:44:42
- Labels: official, publicistic
- Balanced samples per label: 500
- Train size: 800
- Test size: 200
- Max chars per text: 3000

## Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| word_multinomial_nb | 1.0000 | 1.0000 |
| char_tfidf_centroid | 1.0000 | 1.0000 |
| char_multinomial_nb | 0.9950 | 0.9950 |
| word_tfidf_centroid | 0.9950 | 0.9950 |

## Best Model

- Model: word_multinomial_nb
- Accuracy: 1.0000
- Macro F1: 1.0000

## Confusion Matrix

```json
{
  "official": {
    "official": 100,
    "publicistic": 0
  },
  "publicistic": {
    "official": 0,
    "publicistic": 100
  }
}
```

## Top Features

- official: жазылсын, түсімдер, редакцияда, шешіміне, мекемесі, енгізілсін, тізілімінде, қаулының, бекітілсін, желтоқсандағы, етсін, жүктелсін
- publicistic: abai, алынды, сурет, пікір, жаңалықтар, kz, сайтынан, өтті, деді, сонымен, қатар, отыр