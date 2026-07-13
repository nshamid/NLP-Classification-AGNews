# 🧠 NLP Text Classification on AG News Dataset

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BERT-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A comparative study of **Naive Bayes**, **Support Vector Machine (SVM)**, and **BERT base uncased** for multi-class text classification on the [AG News](https://huggingface.co/datasets/wangrongsheng/ag_news) dataset, complete with an interactive Streamlit dashboard for performance benchmarking.

> 📚 Final Project — Natural Language Processing  
> Informatics Engineering · Faculty of Computer Science · Universitas Sriwijaya · 2026

---

## 📊 Results at a Glance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 89.17% | 89.12% | 89.17% | 89.12% |
| SVM | 91.30% | 91.30% | 91.30% | 91.29% |
| **BERT base uncased** | **94.50%** | **94.52%** | **94.50%** | **94.50%** |

---

## 📁 Repository Structure
```text
NLP-Classification-AGNews/
│
├── App/
│   ├── app.py                          # Streamlit dashboard (main app)
│   └── requirements.txt                # Python dependencies
│
├── Images/
│   ├── Confusion Matrix - BERT.png
│   ├── Confusion Matrix - NB.png
│   ├── Confusion Matrix - SVM.png
│   └── README.md
│
├── Model/
│   ├── BERT/                           # Saved BERT fine-tuned model
│   ├── NB/                             # Saved Naive Bayes model & vectorizer
│   └── SVM/                            # Saved SVM model & vectorizer
│
├── Notebook/
│   ├── Naive_Bayes.ipynb
│   └── SVM_Classification_AG_News.ipynb
│
└── README.md
````

---

## 🗂️ Dataset

**AG News Corpus** — a benchmark dataset for topic classification in NLP.

| Property | Detail |
|----------|--------|
| Source | [Hugging Face Datasets](https://huggingface.co/datasets/wangrongsheng/ag_news) |
| Train Set | 120,000 samples (30,000 × 4 classes) |
| Test Set | 7,600 samples (1,900 × 4 classes) |
| Classes | World · Sports · Business · Sci/Tech |
| Balance | Perfectly balanced (25% each) |
| Input | Title + Description concatenated |

---

## 🤖 Models

### 1. 📊 Naive Bayes (Baseline)
- **Algorithm:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF (`max_features=50,000`)
- **Smoothing:** Laplace (`alpha=1.0`)
- **Training Time:** ~3 seconds (CPU)
- **Accuracy:** 89.17%

### 2. ⚡ Support Vector Machine
- **Algorithm:** Linear SVM (`LinearSVC`)
- **Vectorizer:** TF-IDF (`max_features=50,000`)
- **Hyperparameter:** `C=1.0`, `max_iter=1000`
- **Training Time:** ~45 seconds (CPU)
- **Accuracy:** 91.30%

### 3. 🧠 BERT base uncased
- **Model:** `bert-base-uncased` (Hugging Face)
- **Tokenizer:** WordPiece (`max_length=128`)
- **Fine-tuning:** `lr=2e-5`, `epochs=3`, `batch_size=32`
- **Training Time:** ~45 minutes (GPU)
- **Accuracy:** 94.50%

---

## 📈 Confusion Matrices

<table>
  <tr>
    <td align="center"><b>Naive Bayes</b></td>
    <td align="center"><b>SVM</b></td>
    <td align="center"><b>BERT</b></td>
  </tr>
  <tr>
    <td><img src="Images/Confusion Matrix - NB.png" width="280"/></td>
    <td><img src="Images/Confusion Matrix - SVM.png" width="280"/></td>
    <td><img src="Images/Confusion Matrix - BERT.png" width="280"/></td>
  </tr>
</table>

---

## 🖥️ Dashboard

An interactive multi-page Streamlit dashboard is included for visual performance comparison.
(https://classification-agnews.streamlit.app/)

**Pages:**
- 🏠 **Overview** — Model ranking, radar chart, summary table
- 📊 **Metrics Comparison** — Grouped bar chart, per-metric deep-dive
- 🔢 **Confusion Matrix** — Interactive heatmap (raw & normalized)
- 📈 **Per-Class Analysis** — F1-score breakdown per category
- 🔍 **Model Details** — Architecture info, speed vs accuracy trade-off
- ℹ️ **About** — Dataset info, team, and course details

---

## 👥 Development Team

**Group 4 — Informatics Engineering, Universitas Sriwijaya**

| Name | Student ID |
|------|-----------|
| Andini Marsha Daniswara | 09021282328033 |
| Fransisca Stevanie Ekawati | 09021382328127 |
| Nabilah Shamid | 09021382328147 |
| Indrina Nur Chairunnisya | 09021382328157 |
| Shalaisya Fattiha Ramadhani | 09021382328161 |
| Afny Chiara Wildani Nst | 09021382328167 |

**Course Lecturer:** Novi Yusliani, S.Kom, M.T.  
**Course:** Natural Language Processing  
**Institution:** Faculty of Computer Science, Universitas Sriwijaya  
**Year:** 2026

---

<div align="center">
  <sub>NLP Final Project · AG News Text Classification · Universitas Sriwijaya 2026</sub>
</div>
