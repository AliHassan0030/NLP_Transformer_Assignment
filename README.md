# Transformer-Based Sentiment Analysis
## Amazon Polarity | BERT Fine-tuning | SHAP & LIME Explainability

---

## Overview

This project fine-tunes a BERT model for binary sentiment classification on the Amazon Polarity dataset. It includes attention visualization across multiple layers and heads, and applies both SHAP and LIME to explain individual predictions. A comparative analysis evaluates the two explainability methods on runtime, stability, and faithfulness.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | 92.55% |
| Weighted F1 | 0.9255 |
| SHAP samples explained | 25 |
| LIME samples explained | 25 |
| SHAP avg runtime/sample | 6.128s |
| LIME avg runtime/sample | 2.281s |
| LIME Stability (Jaccard) | 0.250 |
| Error rate | 7.45% |

---

## Project Structure

```
nlp-assignment-4/
├── nlp-assignment-4.ipynb       # Main notebook (all code)
├── report/
│   └── NLP_Assignment4_Report.docx
├── outputs/
│   ├── best_bert_model.pt
│   ├── eda_distribution.png
│   ├── methodology_flowchart.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── attn_layer1_head1.png
│   ├── attn_layer1_head2.png
│   ├── attn_layer6_head3.png
│   ├── attn_layer11_head6.png
│   ├── cls_attention_layers.png
│   ├── lime_feature_importance.png
│   ├── test_predictions.csv
│   └── misclassified_samples.csv
└── README.md
```

---

## Dataset

- **Source:** [Amazon Polarity on HuggingFace](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- **Full size:** 3.6M train / 400K test
- **Used subset:** 20K train / 4K val / 4K test (balanced, 10K per class in train)
- **Classes:** Negative (0), Positive (1)

---

## Setup

### Requirements

```
Python 3.10+
PyTorch 2.0+
transformers
datasets
shap
lime
scikit-learn
pandas
numpy
matplotlib
seaborn
bertviz
accelerate
```

### Install Dependencies

```bash
pip install transformers datasets shap lime bertviz accelerate scikit-learn seaborn
```

---

## How to Run

### Option 1: Kaggle (Recommended)

1. Upload `nlp-assignment-4.ipynb` to Kaggle
2. Enable **GPU T4 x2** accelerator
3. Run all cells top to bottom (`Run All`)
4. Output files are saved to `/kaggle/working/`

### Option 2: Local

```bash
# Clone or download the notebook
jupyter notebook nlp-assignment-4.ipynb
```

> Note: Training on CPU is very slow. A GPU with at least 8GB VRAM is recommended.

---

## Notebook Cell Structure

| Cell | Description |
|------|-------------|
| 1 | Environment setup and library installation |
| 2 | Imports and device setup |
| 3 | Hardware and software environment documentation |
| 4 | Dataset loading from HuggingFace |
| 5 | EDA — label distribution and text length analysis |
| 6 | Data sampling and preprocessing |
| 7 | Tokenization with BertTokenizerFast |
| 8 | Methodology flowchart |
| 9 | BERT model definition |
| 10 | Training setup (optimizer, scheduler) |
| 11 | Training and validation functions |
| 12 | Training loop (3 epochs) |
| 13 | Training history plot |
| 14 | Test set evaluation — Accuracy, Precision, Recall, F1 |
| 15 | Confusion matrix |
| 16 | Attention helper functions |
| 17 | Attention heatmaps — positive review |
| 18 | Attention heatmaps — negative review |
| 19 | CLS attention profile across all layers |
| 20 | SHAP pipeline setup |
| 21 | SHAP explanations for 25 samples |
| 22 | SHAP visualizations (bar, waterfall) |
| 23 | LIME setup |
| 24 | LIME explanations for 25 samples |
| 25 | LIME visualizations (feature importance) |
| 26 | SHAP vs LIME comparative analysis |
| 27 | Error analysis |
| 28 | Summary of all results |
| 29 | Save all outputs |

---

## Reproducibility

- Random seed is fixed: `SEED = 42`
- All `np.random.seed`, `torch.manual_seed`, and `random.seed` are set at the start
- The best model is saved as `best_bert_model.pt` and reloaded for evaluation
- SHAP and LIME use the same 25 test samples (selected with a fixed seed)

---

## Hardware Used

| Component | Details |
|-----------|---------|
| GPU | Tesla T4 (15.64 GB) |
| OS | Linux 6.6.122+ |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.0.0 |

---

## References

- [Amazon Polarity Dataset](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/text_examples.html)
- [LIME Documentation](https://lime-ml.readthedocs.io/en/latest/)
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
