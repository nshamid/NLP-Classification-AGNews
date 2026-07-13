# 🧠 NLP Text Classification on AG News Dataset

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BERT-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A comparative study of **Naive Bayes**, **Support Vector Machine (SVM)**, and **BERT base uncased** for multi-class text classification on the [AG News](https://huggingface.co/datasets/ag_news) dataset, complete with an interactive Streamlit dashboard for performance benchmarking.

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
