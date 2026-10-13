# Mental Health Detection from Social Media Data Using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-F79939.svg)](https://scikit-learn.org/)

This project presents a comparative analysis for detecting depression and related mental health conditions (such as anxiety, stress, bipolar disorder, personality disorder, and suicidal ideation) from text-based social media posts[cite: 5, 13]. The repository covers a full pipeline ranging from traditional machine learning approaches (Random Forest, K-Nearest Neighbors) to contextual deep learning architectures (BERT, RoBERTa)[cite: 5, 10].

---

## 📌 Project Overview

* **Author:** Berkant KOZAN[cite: 2, 3]
* **Institution:** Gazi University, Graduate School of Natural and Applied Sciences[cite: 2, 3]
* **Dataset:** Curated multi-class dataset of 53,043 text records sourced from Kaggle (*Sentiment Analysis for Mental Health*)[cite: 9, 13].
* **Task:** Multi-class text classification across 7 target categories:
  * `Normal`, `Depression`, `Anxiety`, `Bipolar`, `Suicidal`, `Stress`, `Personality Disorder`[cite: 13]

---

## 🚀 Key Features & Methodology

1. **Handling Class Imbalance:** Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively within training fold cross-validations to prevent data leakage while enhancing sensitivity toward minority classes (e.g., *Suicidal*, *Stress*)[cite: 5, 10].
2. **Text Representation:**
   * **Classical Models:** TF-IDF vectorization (1,500 terms) combined with `SelectKBest` ($\chi^2$ test) for selecting top 500 features[cite: 10, 11].
   * **Transformer Models:** Tokenization via `AutoTokenizer` (`bert-base-uncased` & `roberta-base`), Byte-Pair Encoding (BPE), and dynamic attention masks ($MAX\_LEN = 128$)[cite: 11, 12].
3. **Cross-Validation:** Evaluated using **10-Fold Stratified Cross-Validation** to ensure statistical reliability on imbalanced distributions[cite: 12].

---

## 📊 Experimental Results & Benchmarks

Experiments were conducted on **53,043 sample posts** using an NVIDIA RTX 4050 GPU (6GB VRAM)[cite: 12, 13].

### Performance Comparison

| Model | Architecture / Method | Test Accuracy | Weighted F1-Score | Key Findings |
| :--- | :--- | :---: | :---: | :--- |
| **KNN (Cosine)** | Distance-based | **48.94%** | **0.47** | Suffered heavily from the *Curse of Dimensionality*[cite: 5, 17]. |
| **Random Forest** | Ensemble Learning + TF-IDF | **74.01%** | **0.74** | Good on `Normal` class, but struggled with overlapping vocabulary (`Depression` vs. `Suicidal`)[cite: 5, 15, 16]. |
| **BERT** | `bert-base-uncased` | **83.41%** | **0.83** | Captures contextual nuances; improved high-risk class identification[cite: 5, 19, 20]. |
| **RoBERTa** *(Best)* | `roberta-base` | **84.18%** | **0.84** | **Top Performing.** Dynamic masking effectively reduced noise and maximized class separability[cite: 21, 22, 25]. |

---

## 📈 ROC-AUC Analysis (RoBERTa Model)

The final RoBERTa model demonstrated exceptional separability across classes, achieving a **0.99 Micro-average AUC**[cite: 23]:

* **Normal:** $AUC = 1.00$[cite: 23]
* **Anxiety / Bipolar / Stress / Personality Disorder:** $AUC = 0.99$[cite: 23]
* **Depression / Suicidal:** $AUC = 0.95$[cite: 23]

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.8+[cite: 12]
* **Core Libraries:** `PyTorch`, `Hugging Face Transformers`, `Scikit-learn`, `Imbalanced-learn` (`imblearn`), `NLTK`, `Pandas`, `NumPy`[cite: 12]
* **Hardware Acceleration:** CUDA / NVIDIA GPU[cite: 12]

