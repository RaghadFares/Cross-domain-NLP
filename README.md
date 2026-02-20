# 🧠 Cross-Domain Mental Health Detection on Twitter
### Can a model trained on depression detect suicidal ideation — without ever seeing it?

---

## 📌 Overview

This project investigates one of the most challenging problems in clinical NLP: **cross-domain transfer in mental health text classification**. We train classifiers on depression-related tweets and evaluate them — without any retraining — on a completely separate suicidal ideation dataset.

The central question:

> *Do depression and suicidal ideation share enough linguistic structure that a model can generalise across the two domains?*

We answer this empirically through **three complete experiments** using progressively richer feature representations, running **six classifiers** in each, and evaluating with rigorous methodology including user-level data splitting, stratified cross-validation, and Precision-Recall curve analysis.

---

## 🗂️ Project Structure

```
NLP_Project/
│
├── 📓 Cross_Domain_NLP_Final_Organized.ipynb   ← Main notebook (all experiments)
│
├── cleaned_data/
│   ├── mddl_cleaned.csv                        ← Preprocessed depression tweets
│   └── suicidal_cleaned.csv                    ← Preprocessed suicidal tweets
│
├── tfidf_results/                              ← Experiment A outputs
│   ├── indomain_results.csv
│   ├── crossdomain_results.csv
│   └── *.png  (charts, PR curves)
│
├── roberta_results/                            ← Experiment B outputs
│   ├── indomain_results.csv
│   ├── crossdomain_results.csv
│   ├── X_full.npy  (embeddings — reusable)
│   ├── X_eval.npy
│   └── pr_curves.png
│
├── mentalbert_results/                         ← Experiment C outputs
│   ├── indomain_results.csv
│   ├── crossdomain_results.csv
│   ├── X_full.npy
│   ├── X_eval.npy
│   └── pr_curves.png
│
└── final_comparison/
    ├── threeway_crossdomain.csv                ← All classifiers, all methods
    ├── threeway_indomain.csv
    └── threeway_comparison.png                 ← Main result visualisation
```

---

## 📊 Datasets

| Dataset | Source | Domain | Labels | Repository |
|---|---|---|---|---|
| **MDDL** | Twitter API | Depression | Depressed / Not Depressed | [sunlightsgy/MDDL](https://github.com/sunlightsgy/MDDL) |
| **twitter-suicidal_data** | Twitter | Suicidal Ideation | Suicidal / Not Suicidal | [laxmemerit/twitter-suicidal-intention-dataset]
(https://github.com/laxmimerit/twitter-suicidal-intention-dataset) |

- The MDDL dataset contains tweets from users who self-reported a depression diagnosis, collected within one month of their disclosure tweet
- The suicidal dataset is used **exclusively for evaluation** — never for training
- **Anchor tweets** (the disclosure posts themselves) were removed from MDDL to prevent trivial label leakage

---

## 🔬 Experiments

All three experiments use **identical protocols** — same training data, same evaluation data, same 6 classifiers, same metrics — so results are directly comparable.

### Experiment A — TF-IDF + Traditional Classifiers
Bag-of-words representation. Fast, interpretable, but blind to word meaning.

### Experiment B — RoBERTa Embeddings
`[CLS]` token embeddings from `roberta-base`. Captures semantic meaning beyond surface word frequencies.

### Experiment C — MentalBERT Embeddings
`[CLS]` token embeddings from `mental/mental-bert-base-uncased` — a BERT model pre-trained specifically on mental health forum data.

### Classifiers (run in all three experiments)
`Naive Bayes` · `Logistic Regression` · `SVM (Linear)` · `Random Forest` · `Decision Tree` · `KNN`

---

## 📈 Results

### Cross-Domain Performance (the core challenge)

| Method | Best F1 | Best Recall | Best AP Score |
|---|---|---|---|
| TF-IDF | 0.2028 *(Naive Bayes)* | 0.1259 *(Naive Bayes)* | 0.5989 *(Random Forest)* |
| MentalBERT | 0.6280 *(KNN)* | 0.5200 *(KNN)* | 0.7264 *(LR)* |
| **RoBERTa** | **0.6447** *(KNN)* | **0.5429** *(KNN)* | **0.7545** *(SVM)* |

### Full Cross-Domain Results — All Classifiers × All Methods

| Model | TF-IDF F1 | TF-IDF Rec | RoBERTa F1 | RoBERTa Rec | MentalBERT F1 | MentalBERT Rec |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.2028 | 0.1259 | 0.4347 | 0.2898 | 0.4845 | 0.3519 |
| Random Forest | 0.0998 | 0.0528 | 0.3517 | 0.2196 | 0.3660 | 0.2315 |
| Decision Tree | 0.0964 | 0.0510 | 0.3974 | 0.2954 | 0.3703 | 0.2632 |
| SVM (Linear) | 0.0146 | 0.0073 | 0.4842 | 0.3339 | 0.2764 | 0.1640 |
| Logistic Regression | 0.0119 | 0.0060 | 0.5105 | 0.3615 | 0.4249 | 0.2811 |
| **KNN** | 0.0009 | 0.0005 | **0.6447** | **0.5429** | **0.6280** | **0.5200** |

### In-Domain Performance (TF-IDF, user-level split)

| Model | CV F1 (mean ± std) | Test F1 |
|---|---|---|
| Decision Tree | 0.9978 ± 0.0013 | 0.9987 |
| Random Forest | 0.9979 ± 0.0012 | 0.9987 |
| SVM (Linear) | 0.9973 ± 0.0019 | 0.9980 |
| Logistic Regression | 0.9923 ± 0.0010 | 0.9961 |
| Naive Bayes | 0.9683 ± 0.0076 | 0.9692 |
| KNN | 0.3210 ± 0.0167 | 0.3519 |

---

## 🔍 Key Findings

**1. Cross-domain transfer is severely limited with TF-IDF**
All classifiers experience 79–99% F1 degradation when moving from in-domain to cross-domain evaluation. A vocabulary overlap analysis reveals only **40% shared top-50 words** between the two domains — depression data is dominated by clinical terminology (`diagnosed`, `bipolar`, `ptsd`) while suicidal data uses action-oriented language (`die`, `kill`, `want`, `end`).

**2. Contextual embeddings substantially close the gap**
RoBERTa improves cross-domain Recall from ~0.007 (TF-IDF best) to 0.5429 (KNN+RoBERTa) — a dramatic improvement — confirming that the failure is a vocabulary mismatch problem that semantic representations can partially bridge.

**3. KNN undergoes the most dramatic reversal of any classifier**
KNN goes from **dead last** with TF-IDF (F1: 0.0009, effectively zero) to **best overall** with RoBERTa (F1: 0.6447, Recall: 0.5429). This is explained by the nature of the feature spaces: TF-IDF produces high-dimensional sparse vectors where nearest-neighbour distances are meaningless (curse of dimensionality), while RoBERTa's dense 768-dimensional embeddings produce a semantic space where true nearest neighbours are genuinely similar tweets. This reversal is the most striking individual finding in the study.

**4. Naive Bayes is the most domain-agnostic traditional classifier**
Despite ranking 5th in-domain, Naive Bayes achieves the best cross-domain F1 among TF-IDF classifiers. Its soft probabilistic decisions allow shared emotional vocabulary to activate across domains, while SVM and Logistic Regression learn rigid boundaries that collapse under distribution shift.

**5. Random Forest has the best probability calibration cross-domain (TF-IDF)**
Random Forest achieves AP Score 0.5989 with TF-IDF — nearly double the no-skill baseline — despite only F1 = 0.0998. Its hard predictions fail due to a too-conservative default threshold, but its probability estimates are well-calibrated. Threshold tuning would substantially improve its performance.

**6. RoBERTa outperforms MentalBERT in feature-extraction mode**
Counterintuitively, the general-purpose model wins. RoBERTa's larger pre-training corpus (160GB) and more robust training procedure outweigh MentalBERT's domain specificity when both are used as frozen feature extractors. MentalBERT would likely close this gap with end-to-end fine-tuning.

---

## ⚙️ Methodology

### Why User-Level Splitting?
We split train/test sets at the **user level**, not the tweet level. Tweet-level splitting allows the same user's tweets in both sets — since people have consistent writing styles, models learn to recognise users rather than generalise. User-level splitting gives honest generalisation estimates.

### Why a Dummy Baseline?
A `DummyClassifier(strategy='most_frequent')` is included as a performance floor. Under class imbalance, it achieves F1 = 0.4771 by predicting everything as suicidal (Recall = 1.0). Real models scoring lower F1 are not worse — they are more selective. The meaningful comparison is Recall and AP Score.

### Why AP Score?
F1 at a fixed threshold can hide model behaviour. Average Precision summarises the full Precision-Recall curve across all thresholds, giving a better picture of a model's ability to rank the positive class — critical when the optimal threshold is unknown.

---

## 🚀 Reproducing the Results

### Requirements
```
Python 3.10+
torch
transformers
scikit-learn
pandas
numpy
matplotlib
wordcloud
huggingface_hub
```

Install:
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib wordcloud huggingface_hub
```

### Running the Notebook
1. Open in **Google Colab** with **T4 GPU** runtime (required for Experiments B & C)
2. Mount Google Drive and set `DRIVE_ROOT` to your project folder
3. Run all cells top to bottom — the notebook is designed to run sequentially

> ⚠️ **MentalBERT** requires a HuggingFace access token with gated repo permissions.
> Request access at: https://huggingface.co/mental/mental-bert-base-uncased

### Reloading Embeddings (avoiding re-extraction)
Embeddings are saved as `.npy` files. If the runtime restarts, reload instead of re-extracting:
```python
X_roberta_full = np.load('roberta_results/X_full.npy')
X_roberta_eval = np.load('roberta_results/X_eval.npy')
```

---

## 📚 References

**Models:**
- Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2022). **MentalBERT: Publicly Available Pretrained Language Models for Mental Health Analysis.** *LREC 2022.* https://arxiv.org/abs/2110.15621
- Liu, Y. et al. (2019). **RoBERTa: A Robustly Optimized BERT Pretraining Approach.** https://arxiv.org/abs/1907.11692
- Coppersmith, G., Dredze, M., & Harman, C. (2014). **Quantifying Mental Health Signals in Twitter.** *ACL Workshop on Computational Linguistics and Clinical Psychology.*

**Datasets:**
- Shen, G. et al. **MDDL — Multi-Domain Depression Detection on Twitter.** GitHub: https://github.com/sunlightsgy/MDDL
- Kumar, L. **Twitter Suicidal Intention Dataset.** GitHub: https://github.com/laxmemerit/twitter-suicidal-intention-dataset

---

## ⚠️ Ethical Considerations

- **This system is not deployment-ready.** The best Recall achieved (0.5429) means roughly 4–5 in every 10 suicidal tweets are still missed. No automated system with this miss rate should be deployed in a real mental health screening context without human-in-the-loop review.
- All data used is from public Twitter posts. No personally identifiable information is stored or processed beyond what is present in the original datasets.
- Any future deployment should prioritise Recall over Precision — a missed suicidal user is more costly than a false alarm.

---

## 👥 Team

- Raghad Fares Almutairi

IT469 – Human Language Technologies
King Saud University · College of Computer and Information Sciences · Spring 2026

---

*For questions about the notebook structure or methodology, refer to the Technical Summary document.*
