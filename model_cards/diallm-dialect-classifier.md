---
language:
- en
license: apache-2.0
tags:
- text-classification
- dialect
- english
- deberta
datasets:
- surrey-nlp/BESSTIE-CW-26
metrics:
- accuracy
- f1
pipeline_tag: text-classification
---

# DiaLLM Dialect Classifier

A fine-tuned [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) model for classifying English text into three dialect varieties: **en-AU** (Australian), **en-IN** (Indian), and **en-UK** (British).

Trained as part of the DiaLLM project — a study of dialect-adapted language models using CPT, SFT, DPO, GRPO, and GSPO across Gemma, Llama, and Qwen model families. Used as an independent evaluation metric to assess whether generated text exhibits target-dialect characteristics.

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="jordanpainter/diallm-dialect-classifier",
)

classifier("I reckon it's a ripper idea, mate.")
# [{'label': 'en-AU', 'score': 0.87}]
```

Labels: `en-AU`, `en-IN`, `en-UK`.

## Training Data

Fine-tuned on [BESSTIE-CW-26](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26), a dataset of 6,243 naturally occurring English sentences annotated for dialect variety. All splits were pooled and re-split 80/10/10 with stratification to ensure balanced dialect representation in dev and test.

| Split | en-AU | en-IN | en-UK | Total |
|-------|-------|-------|-------|-------|
| Train | ~1,619 | ~1,973 | ~1,693 | ~5,285 |
| Val   | ~202  | ~246  | ~211  | ~659  |
| Test  | 192   | 234   | 201   | 627   |

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | microsoft/deberta-v3-base |
| Epochs | 5 (early stopping, patience 2) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max length | 512 |
| Hardware | 1× NVIDIA A100 |

## Evaluation

Test-set results (627 examples, stratified):

| Dialect | Precision | Recall | F1 |
|---------|-----------|--------|----|
| en-AU   | 0.6808    | 0.7552 | 0.7160 |
| en-IN   | 0.8982    | 0.8675 | 0.8826 |
| en-UK   | 0.7234    | 0.6766 | 0.6992 |
| **macro avg** | **0.7675** | **0.7664** | **0.7660** |
| **accuracy** | | | **0.7719** |

Indian English is the most separable class; Australian and British English share substantial lexical overlap, leading to some inter-class confusion between the two.

## Limitations

- Trained on BESSTIE-CW-26, which contains shorter, naturally occurring sentences — performance may vary on longer generated text.
- Confusion between en-AU and en-UK is expected given their shared orthographic conventions.
- Not intended for high-stakes dialect identification; best used as a soft signal in aggregate across many examples.
