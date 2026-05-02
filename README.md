<div align="center">

# 🧠 Mental Health AI Analysis System

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20VADER-FF6B6B?style=for-the-badge&logo=buffer&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TextBlob](https://img.shields.io/badge/TextBlob-Sentiment-4ECDC4?style=for-the-badge&logo=databricks&logoColor=white)](#)
[![Dataset](https://img.shields.io/badge/Dataset-52%2C681%20Posts-A855F7?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-✅%20Complete-2ECC71?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](#)

<br/>

> **52,681 real social media posts** · **7 mental health conditions** · **5 ML models** · **0–100 severity scoring** · **crisis detection** · **coping strategy engine**

<br/>

<div align="center">

### 🔬 See It In Action

| | |
|---|---|
| 📝 **Input Text** | *"I haven't left my bed in 4 days. Everything feels pointless. I don't see a way forward..."* |
| 🔴 **Detected Condition** | `Depression` — confidence 74.3% |
| 📊 **Severity Score** | `68 / 100` &nbsp;🔴 HIGH RISK |
| 💬 **Sentiment** | Negative &nbsp;·&nbsp; VADER: `-0.82` &nbsp;·&nbsp; Subjectivity: `0.71` |
| ⚠️ **Triggers Found** | `Hopelessness` &nbsp;·&nbsp; `Isolation` &nbsp;·&nbsp; `Depression Signs` |
| 💊 **Action** | 4 personalized coping strategies + crisis helpline provided |

</div>

</div>


---



## 🎯 The Problem

Mental health is **one of the most underdiagnosed crises** of our generation.

<div align="center">

### 📊 Global Mental Health Reality

| 🔢 Statistic | 💡 Fact | 😟 Impact |
|-------------|---------|----------|
| **1 in 4 people** | Experience a mental health issue every year | That's ~2 billion people globally |
| **75% of cases** | Go completely undetected or untreated | Millions suffer in silence |
| **800,000+** | Die by suicide annually worldwide | 1 death every 40 seconds |
| **11 years** | Average delay between first symptoms and treatment | A decade of unnecessary suffering |
| **#1 cause** | Mental illness = leading cause of disability worldwide | Bigger than cancer or heart disease |

</div>

People pour their genuine feelings into social media posts **before ever speaking to a doctor**. This project uses AI to detect distress signals in that text — enabling faster intervention, better support systems, and a deeper understanding of mental health at scale.

> *This is not just a machine learning project. It's a tool that could save lives.*

---

## ✨ What Makes This Unique

Most mental health NLP projects stop at "classify the post." This system goes **7 layers deeper:**

```
TYPICAL PROJECT          THIS PROJECT
───────────────          ────────────────────────────────────────────
Predict label      →     ✅ Predict label (7 conditions)
                   →     ✅ Compute severity score (0–100)
                   →     ✅ Dual sentiment engine (VADER + TextBlob)
                   →     ✅ Trigger word & crisis signal detection
                   →     ✅ Personalized coping strategy recommender
                   →     ✅ Automatic crisis alert + real helplines
                   →     ✅ Risk level categorization (5 tiers)
                   →     ✅ Confidence scores for all 7 conditions
```

<br/>

### 🏗️ The 8 Core Features

| # | Feature | Description | Tech Used |
|---|---------|-------------|-----------|
| 1 | 🎯 **7-Class Classifier** | Identifies Normal, Depression, Anxiety, Suicidal, Bipolar, Stress, Personality Disorder | TF-IDF + LinearSVC |
| 2 | 📊 **Severity Scoring** | Assigns a 0–100 risk score based on keywords, labels and sentiment combined | Custom algorithm |
| 3 | 💬 **Dual Sentiment Engine** | VADER for social media tone + TextBlob for polarity and subjectivity | NLTK VADER + TextBlob |
| 4 | ⚠️ **Trigger Word Detector** | Identifies 40+ crisis-level keywords across 4 trigger categories | Rule-based NLP |
| 5 | 💊 **Coping Strategy Engine** | Recommends personalized strategies per detected condition | Knowledge base |
| 6 | 🚨 **Crisis Alert System** | Auto-flags high-risk posts with verified Indian crisis helplines | Threshold logic |
| 7 | 📈 **Confidence Breakdown** | Shows probability % for all 7 conditions — not just top prediction | ML predict_proba |
| 8 | 🔮 **Full Analysis Pipeline** | One function call → complete mental health assessment report | Integrated system |

---

## 📊 Dataset

<div align="center">

### 📦 Dataset At A Glance

| 🏷️ Property | 📋 Details |
|-------------|-----------|
| **Name** | Sentiment Analysis for Mental Health |
| **Source** | Reddit communities · Twitter · Online forums |
| **Total Posts** | 52,681 real posts |
| **Columns** | `statement` (text) · `status` (label) |
| **Avg Post Length** | 578 characters |
| **File Size** | ~85 MB |
| **Preprocessing** | 362 null rows removed |
| **Time Period** | Multi-year collection |

</div>

### 📋 Label Distribution

```
Mental Health Condition    Posts     Share   Visual Distribution
──────────────────────────────────────────────────────────────────
😊  Normal               16,343    31.0%   ████████████████████████
😔  Depression           15,404    29.2%   ██████████████████████
⚠️  Suicidal             10,652    20.2%   ████████████████
😰  Anxiety               3,841     7.3%   ██████
🔄  Bipolar               2,777     5.3%   ████
😤  Stress                2,587     4.9%   ████
🧠  Personality Disorder  1,077     2.0%   ██
──────────────────────────────────────────────────────────────────
    TOTAL                52,681   100.0%
```

### 📁 Dataset Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `statement` | `string` | Raw text post (avg 578 chars) | *"I haven't slept in days..."* |
| `status` | `string` | Mental health category label | `Depression` |
| `label_id` | `int` | Encoded label (0–6) | `1` |

---


### Feature Engineering at a Glance

| Feature Group | Features | Count |
|---------------|----------|-------|
| TF-IDF Unigrams | Top single words | 25,000 |
| TF-IDF Bigrams | 2-word phrases like "panic attack" | included |
| VADER Scores | compound, positive, negative, neutral | 4 |
| TextBlob | polarity, subjectivity | 2 |
| Text Stats | length, word count | 2 |
| **Total** | | **~25,008** |

---

## 🤖 ML Models Compared

Five models trained and evaluated head-to-head:

| Model | Type | Strength | Class Weight |
|-------|------|----------|-------------|
| 🔵 Naive Bayes | Probabilistic | Fast baseline, great with text | — |
| 🟢 Logistic Regression | Linear | Interpretable, strong with TF-IDF | ✅ Balanced |
| 🔴 **Linear SVM** | **Margin-based** | **Best for high-dimensional text** | ✅ Balanced |
| 🟡 Random Forest | Ensemble | Handles non-linear patterns | ✅ Balanced |
| 🟣 Gradient Boosting | Boosting | Handles complex decision boundaries | — |

> All models trained on **42,144 posts** (80%) · Evaluated on **10,537 posts** (20%) · Stratified split

---

## 📈 Results & Performance

<div align="center">

### 🏆 Model Leaderboard

| Rank | Model | Accuracy | Weighted F1 | Macro F1 | Verdict |
|------|-------|----------|-------------|----------|---------|
| 🥇 | **Linear SVM** | **~86%** | **~85%** | **~78%** | ✅ Best Overall |
| 🥈 | Logistic Regression | ~85% | ~84% | ~76% | ✅ Great Runner-up |
| 🥉 | Gradient Boosting | ~83% | ~82% | ~74% | 👍 Strong |
| 4️⃣ | Random Forest | ~81% | ~80% | ~71% | 👍 Good |
| 5️⃣ | Naive Bayes | ~72% | ~71% | ~63% | ⚡ Fast Baseline |

> 🧪 Trained on **42,144 posts** · Tested on **10,537 posts** · Stratified 80/20 split · Class-balanced weights applied

</div>

### 🔬 Per-Class Performance — Best Model

| Condition | Precision | Recall | F1-Score | Test Posts |
|-----------|-----------|--------|----------|------------|
| 😊 Normal | ~91% | ~93% | ~92% | 3,269 |
| 😔 Depression | ~88% | ~90% | ~89% | 3,081 |
| ⚠️ Suicidal | ~90% | ~88% | ~89% | 2,130 |
| 😰 Anxiety | ~78% | ~75% | ~76% | 768 |
| 🔄 Bipolar | ~74% | ~71% | ~72% | 555 |
| 😤 Stress | ~73% | ~70% | ~71% | 517 |
| 🧠 Personality Disorder | ~68% | ~65% | ~66% | 215 |

### 📊 Severity Score System — 5 Risk Tiers

```
Risk Tier    Score Range    Meaning                  Action
─────────────────────────────────────────────────────────────────
🟢 LOW        0  –  20     No significant signals   Monitor
🟡 MILD       21 –  40     Mild distress signals    Acknowledge
🟠 MODERATE   41 –  60     Moderate concern         Recommend support
🔴 HIGH       61 –  80     High risk signals        Urgent support
🚨 CRISIS     81 – 100     Crisis-level signals     Immediate help
```

---

## 🗂️ Project Structure

```
📁 mental-health-ai-system/
│
├── 📊 data/
│   └── mental_health_combined.csv       # 52,681 real posts
│
├── 📓 notebooks/
│   └── Mental_Health_AI_System.py       # Complete Colab-ready code
│
├── 🤖 models/
│   ├── mental_health_model.pkl          # Trained classifier
│   ├── mental_health_tfidf.pkl          # TF-IDF vectorizer
│   └── mental_health_encoder.pkl        # Label encoder
│
├── 📸 charts/
│   ├── chart1_overview.png              # Class distribution
│   ├── chart2_sentiment_analysis.png    # VADER + TextBlob deep dive
│   ├── chart3_wordclouds.png            # Per-condition word clouds
│   ├── chart4_trigger_analysis.png      # Trigger word detection rates
│   ├── chart5_top_keywords.png          # Most distinctive words
│   ├── chart6_model_comparison.png      # 5-model benchmark
│   ├── chart7_confusion_matrix.png      # Raw + % confusion matrix
│   ├── chart8_per_class_metrics.png     # Precision / Recall / F1
│   └── chart9_severity_analysis.png     # Risk score distribution
│
├── 📄 README.md                         # This file
└── 📋 requirements.txt                  # All dependencies
```

---


**Sample Output:**

```
════════════════════════════════════════════════════════════
  🧠  MENTAL HEALTH AI ANALYSIS SYSTEM
════════════════════════════════════════════════════════════

  📊 PREDICTION
  Detected Condition : Depression
  Severity Score     : 63 / 100
  Risk Level         : 🔴 HIGH RISK

  📈 CONFIDENCE SCORES (all conditions):
  Depression               72.3%  ██████████████
  Normal                   12.1%  ██
  Anxiety                   8.4%  █
  Stress                    4.2%  █
  Bipolar                   2.1%
  Suicidal                  0.7%
  Personality disorder      0.2%

  💬 SENTIMENT ANALYSIS
  Overall Sentiment  : Negative 😔
  VADER Score        : -0.741
  Subjectivity       : 0.682

  ⚠️  TRIGGER SIGNALS DETECTED:
  • Depression Signs
  • Hopelessness Indicators

  💊 RECOMMENDED COPING STRATEGIES:
  🌅 Maintain a consistent wake-up time — routine helps depression
  🤝 Reach out to one person today, even a short message counts
  🏃 Even 10 minutes of movement shifts brain chemistry positively
  📞 iCall (India): 9152987821 — free mental health support

════════════════════════════════════════════════════════════
```

---

## 💼 Business & Social Impact

This system has **real-world applications** across multiple industries:

```
🏥  Healthcare Platforms     →  Early detection and patient triage
📱  Social Media Companies   →  Content moderation and user safety
🎓  Universities             →  Student mental wellness monitoring
💼  HR / Corporates          →  Employee burnout early warning system
🌐  NGOs / Helplines         →  Automate screening at scale
🔬  Researchers              →  Population-level mental health trends
```

### 💰 Why This Matters to Recruiters

> Mental health AI is a **$4.2 billion market** growing at **24% annually**. Companies like **Woebot, Wysa, Calm** and **BetterHelp** all use NLP-powered systems similar to this as their core product technology.

---

## 🧰 Full Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **NLP** | NLTK (VADER, lemmatization, stopwords), TextBlob, TF-IDF |
| **ML Models** | Scikit-learn (NB, LR, SVM, RF, GBM) |
| **Imbalance Handling** | imbalanced-learn (class weighting) |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Environment** | Google Colab / Local Python |
| **Version Control** | Git + GitHub |

---

## 👨‍💻 Author

<div align="center">

**Arushi Garg**

*B.Tech Computer Science (AI and Data Science)*

[![Email](https://img.shields.io/badge/Email-Arushigarg525@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Arushigarg525@email.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shinearushi525)

</div>

---

<div align="center">

</div>
