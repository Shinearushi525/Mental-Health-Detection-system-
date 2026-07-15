"""
Mental Health AI Analysis System — Streamlit App
Classifies a piece of text into one of 7 mental-health categories,
computes a 0-100 severity/risk score, detects trigger words, runs
sentiment analysis, and recommends coping strategies / crisis resources.

NOTE: This is an educational NLP project, not a diagnostic or clinical
tool. It must never be used as a substitute for professional help.
"""
import json
import pickle
import re

import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health AI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# One-time setup: NLTK data + cached artifact loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up NLP resources...")
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "vader_lexicon", "punkt"]:
        nltk.download(pkg, quiet=True)
    return True


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    with open("mental_health_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("mental_health_tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("mental_health_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    try:
        with open("model_info.json", "r") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}
    return model, tfidf, encoder, info


setup_nltk()
MODEL, TFIDF, ENCODER, MODEL_INFO = load_artifacts()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
SIA = SentimentIntensityAnalyzer()

# --------------------------------------------------------------------------
# Domain logic (mirrors the training notebook)
# --------------------------------------------------------------------------
TRIGGER_KEYWORDS = {
    "Suicidal Risk": ["suicide", "kill myself", "end my life", "want to die",
                      "no reason to live", "better off dead", "self harm",
                      "cut myself", "overdose", "worthless", "hopeless"],
    "Depression Signs": ["empty", "numb", "pointless", "exhausted", "crying",
                          "isolate", "no energy", "sleep all day", "worthless",
                          "no motivation", "can't get up"],
    "Anxiety Signs": ["panic attack", "heart racing", "can't breathe",
                       "overthinking", "catastrophize", "worry", "dread",
                       "nervous", "trembling", "shaking"],
    "Crisis Words": ["crisis", "emergency", "help me", "please help",
                      "desperate", "can't take it", "breaking down"],
}

LABEL_BASE_SEVERITY = {
    "Normal": 5, "Stress": 25, "Anxiety": 40,
    "Depression": 55, "Bipolar": 60,
    "Personality disorder": 65, "Suicidal": 85,
}

PALETTE = {
    "Normal": "#2ECC71", "Depression": "#3498DB", "Suicidal": "#E74C3C",
    "Anxiety": "#F39C12", "Bipolar": "#9B59B6", "Stress": "#E67E22",
    "Personality disorder": "#1ABC9C",
}

RISK_LEVELS = {
    (0, 20): ("LOW", "#2ECC71", "🟢"),
    (21, 40): ("MILD", "#F1C40F", "🟡"),
    (41, 60): ("MODERATE", "#E67E22", "🟠"),
    (61, 80): ("HIGH", "#E74C3C", "🔴"),
    (81, 100): ("CRISIS", "#8E0000", "🚨"),
}

COPING_STRATEGIES = {
    "Normal": [
        "You seem to be doing well! Keep maintaining healthy habits.",
        "Continue your current self-care routine.",
        "Consider helping others — it boosts your own well-being too.",
    ],
    "Stress": [
        "Try 5-minute deep breathing: inhale 4s, hold 4s, exhale 6s.",
        "Write down your top 3 stressors and one small step for each.",
        "A 15-minute walk can reduce cortisol levels significantly.",
        "Set a phone-free hour before bed to decompress.",
    ],
    "Anxiety": [
        "The 5-4-3-2-1 grounding technique: name 5 things you see, 4 you feel, 3 you hear.",
        "Challenge anxious thoughts: is this fear realistic? What's the evidence?",
        "Reduce caffeine intake — it can amplify anxiety symptoms.",
        "Consider speaking to a therapist who specializes in CBT for anxiety.",
    ],
    "Depression": [
        "Try to maintain a consistent wake-up time — routine helps with depression.",
        "Reach out to one person today, even if just a short message.",
        "Even 10 minutes of movement can shift brain chemistry positively.",
        "iCall helpline (India): 9152987821 — free mental health support.",
    ],
    "Bipolar": [
        "Track your mood daily — apps like eMoods help spot patterns early.",
        "Sleep consistency is critical for bipolar — aim for the same schedule.",
        "Never adjust medication without consulting your psychiatrist.",
        "Share your mood chart with a trusted person who can flag warning signs.",
    ],
    "Personality disorder": [
        "DBT (Dialectical Behavior Therapy) is highly effective — ask about it.",
        "Practice TIPP skills: Temperature, Intense exercise, Paced breathing.",
        "iCall helpline: 9152987821 for professional guidance.",
    ],
    "Suicidal": [
        "IMMEDIATE: iCall (India) — 9152987821 (Mon–Sat, 8am–10pm)",
        "Vandrevala Foundation: 1860-2662-345 (24/7, free)",
        "AASRA: 9820466627 (24/7)",
        "You matter. Please reach out to someone you trust right now.",
        "Text a trusted friend or family member immediately.",
    ],
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def get_sentiment_scores(text: str) -> dict:
    vader = SIA.polarity_scores(str(text))
    blob = TextBlob(str(text))
    return {
        "vader_compound": vader["compound"],
        "vader_negative": vader["neg"],
        "vader_positive": vader["pos"],
        "vader_neutral": vader["neu"],
        "tb_polarity": blob.sentiment.polarity,
        "tb_subjectivity": blob.sentiment.subjectivity,
    }


def compute_severity_score(text: str, label: str) -> int:
    text_lower = str(text).lower()
    score = LABEL_BASE_SEVERITY.get(label, 30)
    for word in TRIGGER_KEYWORDS["Suicidal Risk"]:
        if word in text_lower:
            score += 5
    for word in TRIGGER_KEYWORDS["Depression Signs"]:
        if word in text_lower:
            score += 2
    for word in TRIGGER_KEYWORDS["Crisis Words"]:
        if word in text_lower:
            score += 3
    vader = SIA.polarity_scores(text_lower)
    if vader["compound"] < -0.5:
        score += 10
    elif vader["compound"] < -0.2:
        score += 5
    return min(int(score), 100)


def detect_triggers(text: str) -> list:
    text_lower = str(text).lower()
    found = []
    for category, words in TRIGGER_KEYWORDS.items():
        if any(w in text_lower for w in words):
            found.append(category)
    return found if found else ["None detected"]


def get_risk_level(score: int):
    for (low, high), val in RISK_LEVELS.items():
        if low <= score <= high:
            return val
    return RISK_LEVELS[(0, 20)]


def analyze(text: str) -> dict:
    cleaned = clean_text(text)
    tfidf_vec = TFIDF.transform([cleaned])
    sentiment = get_sentiment_scores(text)
    extras = np.array([[
        len(text), len(text.split()),
        sentiment["vader_compound"], sentiment["vader_negative"],
        sentiment["vader_positive"], sentiment["tb_polarity"],
        sentiment["tb_subjectivity"],
    ]])
    X_input = hstack([tfidf_vec, csr_matrix(extras)])

    pred_enc = MODEL.predict(X_input)[0]
    predicted_label = ENCODER.inverse_transform([pred_enc])[0]

    # Confidence: use predict_proba if available, otherwise derive a
    # relative confidence from decision_function via softmax.
    confidences = None
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X_input)[0]
        labels = ENCODER.inverse_transform(range(len(proba)))
        confidences = dict(zip(labels, proba))
    elif hasattr(MODEL, "decision_function"):
        scores = MODEL.decision_function(X_input)[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        labels = ENCODER.inverse_transform(range(len(proba)))
        confidences = dict(zip(labels, proba))

    sev_score = compute_severity_score(text, predicted_label)
    risk_label, risk_color, risk_emoji = get_risk_level(sev_score)
    triggers = detect_triggers(text)

    vader_score = sentiment["vader_compound"]
    if vader_score > 0.05:
        polarity = "Positive 😊"
    elif vader_score < -0.05:
        polarity = "Negative 😔"
    else:
        polarity = "Neutral 😐"

    return {
        "condition": predicted_label,
        "confidences": confidences,
        "severity": sev_score,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "risk_emoji": risk_emoji,
        "polarity": polarity,
        "vader_score": vader_score,
        "subjectivity": sentiment["tb_subjectivity"],
        "triggers": triggers,
        "is_crisis": predicted_label == "Suicidal" or sev_score >= 80,
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🧠 Mental Health AI Analysis System")
st.caption(
    "An NLP system that classifies text into mental-health categories, "
    "scores severity/risk, and suggests coping strategies."
)

with st.sidebar:
    st.header("About this project")
    if MODEL_INFO:
        st.metric("Best model", MODEL_INFO.get("best_model", "—"))
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{MODEL_INFO.get('accuracy', '—')}%")
        c2.metric("Weighted F1", f"{MODEL_INFO.get('weighted_f1', '—')}%")
        st.caption(f"Trained on {MODEL_INFO.get('n_posts', '—'):,} posts across "
                   f"{len(MODEL_INFO.get('labels', []))} categories.")
        with st.expander("All models compared"):
            for r in MODEL_INFO.get("all_results", []):
                st.write(f"**{r['Model']}** — Acc: {r['Accuracy']*100:.1f}% · "
                         f"Weighted F1: {r['Weighted F1']*100:.1f}%")
    st.divider()
    st.subheader("🆘 Crisis helplines (India)")
    st.write("- iCall: 9152987821 (Mon–Sat, 8am–10pm)")
    st.write("- Vandrevala Foundation: 1860-2662-345 (24/7)")
    st.write("- AASRA: 9820466627 (24/7)")
    st.caption("Outside India, search for a local crisis line or contact "
               "emergency services.")

example_posts = {
    "— Select an example —": "",
    "Suicidal": "I can't take it anymore. I have no reason to live. Every day is just pain. I've been thinking about ending it all.",
    "Anxiety": "My hands won't stop shaking before my presentation tomorrow. I keep imagining every possible way it could go wrong.",
    "Depression": "I haven't left my bed in four days. Everything feels pointless. I used to love painting but now I just stare at the blank canvas.",
    "Stress": "Three project deadlines this week and my manager just added another one. I haven't slept properly in days.",
    "Normal": "Had a great workout today! Feeling strong and motivated. Cooked a healthy meal and caught up with an old friend.",
}

choice = st.selectbox("Try an example, or write your own text below:", list(example_posts.keys()))
default_text = example_posts[choice]

user_text = st.text_area(
    "Enter text to analyze",
    value=default_text,
    height=150,
    placeholder="Type or paste a journal entry, social media post, or message...",
)

analyze_clicked = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

if analyze_clicked:
    if not user_text.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = analyze(user_text)

        if result["is_crisis"]:
            st.error(
                "🚨 **CRISIS ALERT** — This text shows signs of a potential mental "
                "health crisis. If this reflects how you or someone else is "
                "feeling right now, please reach out immediately:\n\n"
                "- **iCall:** 9152987821\n"
                "- **Vandrevala Foundation:** 1860-2662-345 (24/7)\n"
                "- **AASRA:** 9820466627 (24/7)\n\n"
                "You are not alone, and help is available."
            )

        col1, col2, col3 = st.columns(3)
        color = PALETTE.get(result["condition"], "#888")
        col1.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{color}22;"
            f"border:2px solid {color};text-align:center'>"
            f"<div style='font-size:13px;color:#555'>DETECTED CONDITION</div>"
            f"<div style='font-size:24px;font-weight:700;color:{color}'>{result['condition']}</div>"
            f"</div>", unsafe_allow_html=True)
        col2.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{result['risk_color']}22;"
            f"border:2px solid {result['risk_color']};text-align:center'>"
            f"<div style='font-size:13px;color:#555'>RISK LEVEL</div>"
            f"<div style='font-size:24px;font-weight:700;color:{result['risk_color']}'>"
            f"{result['risk_emoji']} {result['risk_label']}</div></div>", unsafe_allow_html=True)
        col3.markdown(
            f"<div style='padding:16px;border-radius:10px;background:#88888822;"
            f"border:2px solid #888;text-align:center'>"
            f"<div style='font-size:13px;color:#555'>SEVERITY SCORE</div>"
            f"<div style='font-size:24px;font-weight:700'>{result['severity']}/100</div>"
            f"</div>", unsafe_allow_html=True)

        st.write("")
        st.progress(result["severity"] / 100, text=f"Severity: {result['severity']}/100")

        st.subheader("📈 Confidence across categories")
        if result["confidences"]:
            sorted_conf = sorted(result["confidences"].items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_conf:
                st.write(f"**{label}** — {prob*100:.1f}%")
                st.progress(float(prob))
        else:
            st.caption("Confidence scores are not available for this model.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💬 Sentiment")
            st.write(f"**Overall sentiment:** {result['polarity']}")
            st.write(f"**VADER compound score:** {result['vader_score']:.3f}")
            st.write(f"**Subjectivity:** {result['subjectivity']:.3f}")
        with c2:
            st.subheader("⚠️ Trigger signals detected")
            for t in result["triggers"]:
                st.write(f"- {t}")

        st.subheader("💊 Recommended coping strategies")
        strategies = COPING_STRATEGIES.get(result["condition"], COPING_STRATEGIES["Normal"])
        for s in strategies:
            st.write(f"- {s}")

st.divider()
st.caption(
    "Built from a 7-class NLP classifier (TF-IDF + linear model) trained on "
    "~53,000 labeled social-media posts. For educational/demo purposes only — "
    "not a substitute for professional mental health care."
)
