import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Emotion Detector", page_icon="🧠", layout="centered")

# ---------- Labels (EDIT to match your dataset) ----------
# If your model predicts text labels (e.g. "joy"), this is ignored.
LABELS = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy",
}
EMOJI = {
    "sadness": "😢", "anger": "😠", "love": "❤️",
    "surprise": "😲", "fear": "😨", "joy": "😄",
}
COLORS = {
    "sadness": "#4A6FA5", "anger": "#D64545", "love": "#E75A97",
    "surprise": "#F2A93B", "fear": "#7A5AA6", "joy": "#2FB37A",
}


# ---------- Load model files once ----------
@st.cache_resource
def load_files():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


def clean_text(text: str) -> str:
    """Replace with the SAME cleaning steps used in your notebook."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def name_of(label):
    return LABELS.get(label, str(label)) if not isinstance(label, str) else label


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .title {text-align:center; font-size:2.4rem; font-weight:800; margin-bottom:0;}
    .subtitle {text-align:center; color:#888; margin-bottom:1.5rem;}
    .result {padding:1.5rem; border-radius:16px; text-align:center; color:white;
             margin-top:1rem; box-shadow:0 4px 14px rgba(0,0,0,.2);}
    .result .emoji {font-size:3.5rem;}
    .result .name {font-size:1.8rem; font-weight:700; text-transform:capitalize;}
    .result .conf {opacity:.9;}
    div.stButton > button {width:100%; border-radius:10px; height:3rem; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🧠 Emotion Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Type a sentence and the model will detect the emotion behind it.</div>',
    unsafe_allow_html=True,
)

try:
    model, vectorizer = load_files()
except FileNotFoundError:
    st.error("Could not find best_model.pkl / vectorizer.pkl. Put them in the same folder as app.py.")
    st.stop()

# ---------- Input ----------
examples = {
    "Choose an example...": "",
    "😄 I am so happy today, everything went great!": "I am so happy today, everything went great!",
    "😢 I feel so lonely and hopeless": "I feel so lonely and hopeless",
    "😠 I am furious about how they treated me": "I am furious about how they treated me",
    "😨 I am terrified of what might happen": "I am terrified of what might happen",
}
choice = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area("Your text", value=examples[choice], height=140,
                    placeholder="Write how you feel...")

if st.button("Analyze emotion", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        X = vectorizer.transform([clean_text(text)])
        pred = model.predict(X)[0]
        name = name_of(pred)
        color = COLORS.get(name, "#555")

        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        conf = f"Confidence: {proba.max() * 100:.1f}%" if proba is not None else ""

        st.markdown(
            f"""
            <div class="result" style="background:{color};">
                <div class="emoji">{EMOJI.get(name, "🙂")}</div>
                <div class="name">{name}</div>
                <div class="conf">{conf}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if proba is not None:
            st.subheader("Probability of each emotion")
            df = pd.DataFrame({
                "Emotion": [name_of(c) for c in model.classes_],
                "Probability": proba,
            }).sort_values("Probability", ascending=False)
            st.bar_chart(df.set_index("Emotion"))

st.caption("Built with Streamlit • TF-IDF + Logistic Regression")
