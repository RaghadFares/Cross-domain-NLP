
# Cross-Domain Mental Health Detection on Twitter
### Can a model trained on depression detect suicidal ideation — without ever seeing it?

---

## Overview

This project investigates cross-domain transfer in mental health text classification. 
We train classifiers on depression-related tweets and evaluate them — without any 
retraining — on a completely separate suicidal ideation dataset.

The central question:

> *Do depression and suicidal ideation share enough linguistic structure that a model 
> can generalise across the two domains?*

We answer this empirically through three complete experiments using progressively 
richer feature representations, running six classifiers in each, and evaluating with 
rigorous methodology including user-level data splitting, stratified cross-validation, 
and Precision-Recall curve analysis.

---

## Repository Structure

```
Cross-Domain-NLP/
│
├── data/
│  
├── notebooks/
│   └── Cross_Domain_NLP_Code.ipynb   # Main notebook — run this
│
├── results/
│   ├── tfidf_results/                 # Experiment A outputs
│   ├── roberta_results/               # Experiment B outputs
│   ├── mentalbert_results/            # Experiment C outputs
│   └── final_comparison/             # Three-way comparison charts
│
├── report/
│   └── IT469_Group8_Report.pdf        # Final written report
│
├── technical_summary/
│   └── Technical_Summary.pdf         # Detailed methodology notes
│
└── README.md
```

---

## Datasets

The datasets are not included in this repository due to size. 
Download them from the links below and place them in the correct folders.

| Dataset | Domain | Labels | Download |
|---|---|---|---|
| MDDL | Depression | Depressed / Not Depressed | [sunlightsgy/MDDL](https://github.com/sunlightsgy/MDDL) |
| twitter-suicidal_data | Suicidal Ideation | Suicidal / Not Suicidal | [laxmemerit/twitter-suicidal-intention-dataset](https://github.com/laxmimerit/twitter-suicidal-intention-dataset) |

- The MDDL dataset contains tweets from users who self-reported a depression 
  diagnosis, collected within one month of their disclosure tweet.
- The suicidal dataset is used exclusively for evaluation — never for training.
- Anchor tweets (the disclosure posts themselves) were removed from MDDL 
  to prevent trivial label leakage.

---

## Experiments

All three experiments use identical protocols — same training data, same evaluation 
data, same 6 classifiers, same metrics — so results are directly comparable.

**Experiment A — TF-IDF + Traditional Classifiers**
Bag-of-words representation. Fast, interpretable, but blind to word meaning.

**Experiment B — RoBERTa Embeddings**
CLS token embeddings from roberta-base. Captures semantic meaning beyond 
surface word frequencies.

**Experiment C — MentalBERT Embeddings**
CLS token embeddings from mental/mental-bert-base-uncased — a BERT model 
pre-trained specifically on mental health forum data.

**Classifiers (run in all three experiments)**
Naive Bayes, Logistic Regression, SVM (Linear), Random Forest, Decision Tree, KNN

---

## Results

### Cross-Domain Performance

| Method | Best F1 | Best Recall | Best AP Score |
|---|---|---|---|
| TF-IDF | 0.2028 (Naive Bayes) | 0.1259 (Naive Bayes) | 0.5989 (Random Forest) |
| MentalBERT | 0.6280 (KNN) | 0.5200 (KNN) | 0.7264 (LR) |
| RoBERTa | 0.6447 (KNN) | 0.5429 (KNN) | 0.7545 (SVM) |

### Full Cross-Domain Results

| Model | TF-IDF F1 | TF-IDF Rec | RoBERTa F1 | RoBERTa Rec | MentalBERT F1 | MentalBERT Rec |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.2028 | 0.1259 | 0.4347 | 0.2898 | 0.4845 | 0.3519 |
| Random Forest | 0.0998 | 0.0528 | 0.3517 | 0.2196 | 0.3660 | 0.2315 |
| Decision Tree | 0.0964 | 0.0510 | 0.3974 | 0.2954 | 0.3703 | 0.2632 |
| SVM (Linear) | 0.0146 | 0.0073 | 0.4842 | 0.3339 | 0.2764 | 0.1640 |
| Logistic Regression | 0.0119 | 0.0060 | 0.5105 | 0.3615 | 0.4249 | 0.2811 |
| KNN | 0.0009 | 0.0005 | 0.6447 | 0.5429 | 0.6280 | 0.5200 |

---

## Key Findings

**1. TF-IDF fails catastrophically under domain shift**
All classifiers experience 79–99% F1 degradation. Only 40% of the top-50 words 
are shared between the two domains — depression data uses clinical terminology 
(diagnosed, bipolar, ptsd) while suicidal data uses action-oriented language 
(die, kill, want, end).

**2. Contextual embeddings substantially close the gap**
RoBERTa improves cross-domain Recall from 0.007 to 0.5429 — confirming that 
the failure is a vocabulary mismatch problem that semantic representations 
can partially bridge.

**3. The KNN Reversal**
KNN goes from dead last with TF-IDF (F1: 0.0009) to best overall with RoBERTa 
(F1: 0.6447). In sparse TF-IDF space, nearest-neighbour distances are meaningless. 
In RoBERTa's dense 768-dimensional space, true nearest neighbours are genuinely 
semantically similar tweets.

**4. RoBERTa outperforms MentalBERT in feature-extraction mode**
The general-purpose model wins. RoBERTa's larger pre-training corpus (160GB) 
outweighs MentalBERT's domain specificity when both are used as frozen 
feature extractors.

---

## How to Run

### Requirements
```
Python 3.10+
torch, transformers, scikit-learn
pandas, numpy, matplotlib
wordcloud, huggingface_hub
```

Install:
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib wordcloud huggingface_hub
```

### Steps
1. Download both datasets and place them in the data/ folder
2. Open Cross_Domain_NLP_Code.ipynb in Google Colab with T4 GPU runtime
3. Mount Google Drive and set DRIVE_ROOT to your project folder
4. Run all cells top to bottom

> MentalBERT requires a HuggingFace access token with gated repo permissions.
> Request access at: https://huggingface.co/mental/mental-bert-base-uncased

### Reloading Embeddings
If the runtime restarts, reload saved embeddings instead of re-extracting:
```python
X_roberta_full = np.load('roberta_results/X_full.npy')
X_roberta_eval = np.load('roberta_results/X_eval.npy')
```

---

## Ethical Considerations

This system is not deployment-ready. The best Recall achieved (0.5429) means 
roughly 4-5 in every 10 suicidal tweets are still missed. No automated system 
with this miss rate should be deployed in a real mental health screening context 
without human-in-the-loop review.

Any future deployment should prioritise Recall over Precision — a missed suicidal 
user is more costly than a false alarm.

---

## Team

Raghad Fares Almutairi, Aljwharah Alhowidy, Mariam Alahmed,
Raghad Sultan Aldajani, Nora Fisal Albyahi

IT469 – Human Language Technologies
King Saud University · Spring 2026
