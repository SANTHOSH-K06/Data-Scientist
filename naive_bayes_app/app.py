import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ---- Page Config ----
st.set_page_config(
    page_title="News Classifier | Naive Bayes",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0f0020 0%, #1a0030 50%, #0a0018 100%);
    color: #f0e8ff;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, rgba(139,0,220,0.35), rgba(100,0,180,0.25));
    border: 1px solid rgba(180,0,255,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 40px rgba(139,0,220,0.2);
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #c084fc, #e879f9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px !important;
}
.hero p { color: rgba(220,180,255,0.75); font-size: 0.95rem; margin: 0; }

.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(192,132,252,0.14);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}
.sec-title {
    font-size: 0.9rem; font-weight: 700; color: #c084fc;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6b21a8, #9333ea) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85em 2em !important;
    box-shadow: 0 6px 20px rgba(147,51,234,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.result-card {
    background: rgba(192,132,252,0.1);
    border: 1px solid rgba(192,132,252,0.3);
    border-left: 5px solid #c084fc;
    border-radius: 16px;
    padding: 26px 22px;
    margin-top: 20px;
    animation: fadeIn 0.5s ease;
}
.big-cat { font-size: 1.8rem; font-weight: 800; color: #c084fc; margin: 8px 0; }

.conf-bar { background: rgba(255,255,255,0.08); border-radius: 50px; height: 10px; margin-top: 12px; overflow: hidden; }
.conf-fill { height: 100%; background: linear-gradient(90deg, #6b21a8, #c084fc); border-radius: 50px; }

.stTextArea label { color: #d8b4fe !important; font-weight: 600 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0015, #070010) !important;
    border-right: 1px solid rgba(192,132,252,0.12) !important;
}
[data-testid="stSidebar"] * { color: #d8b4fe !important; }
.sidebar-card {
    background: rgba(192,132,252,0.07);
    border: 1px solid rgba(192,132,252,0.18);
    border-radius: 12px;
    padding: 13px;
    font-size: 0.84rem;
    line-height: 1.65;
    margin-bottom: 12px;
}
.cat-chip {
    display: inline-block;
    background: rgba(192,132,252,0.12);
    border: 1px solid rgba(192,132,252,0.25);
    color: #d8b4fe;
    border-radius: 50px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer { text-align: center; padding: 16px; color: rgba(180,120,255,0.3); font-size: 0.78rem; margin-top: 30px; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ---- Categories ----
CATEGORIES = [
    'sci.space', 'rec.sport.hockey', 'talk.religion.misc',
    'comp.graphics', 'rec.autos', 'sci.med', 'talk.politics.guns'
]

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 📰 About")
    st.markdown("""<div class="sidebar-card">
    Classifies news articles into <strong>7 categories</strong> using 
    <strong>Multinomial Naive Bayes</strong> + TF-IDF vectorization.
    </div>""", unsafe_allow_html=True)
    alpha = st.slider("Smoothing (α)", 0.01, 2.0, 1.0, 0.01)
    st.markdown("---")
    st.markdown("### 📂 Categories")
    cats_html = "".join([f'<span class="cat-chip">{c}</span>' for c in CATEGORIES])
    st.markdown(f'<div class="sidebar-card">{cats_html}</div>', unsafe_allow_html=True)

# ---- Train ----
@st.cache_data
def train_nb(alpha):
    train_data = fetch_20newsgroups(subset='train', categories=CATEGORIES, remove=('headers','footers','quotes'))
    test_data  = fetch_20newsgroups(subset='test',  categories=CATEGORIES, remove=('headers','footers','quotes'))
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,2))),
        ('nb',    MultinomialNB(alpha=alpha))
    ])
    pipe.fit(train_data.data, train_data.target)
    acc = accuracy_score(test_data.target, pipe.predict(test_data.data))
    return pipe, acc, train_data.target_names

pipe, acc, target_names = train_nb(alpha)

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>📰 News Article Classifier</h1>
    <p>Classify news text into categories using Naive Bayes and TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ---- Metrics ----
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="glass"><div class="sec-title">🎯 Accuracy</div><div style="font-size:1.8rem;font-weight:800;color:#c084fc;">{acc*100:.2f}%</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="glass"><div class="sec-title">📂 Categories</div><div style="font-size:1.8rem;font-weight:800;color:#c084fc;">{len(CATEGORIES)}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="glass"><div class="sec-title">🔤 Smoothing α</div><div style="font-size:1.8rem;font-weight:800;color:#c084fc;">{alpha}</div></div>', unsafe_allow_html=True)

# ---- Text Input ----
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📝 Enter News Text</div>', unsafe_allow_html=True)

example_texts = {
    "Science / Space": "NASA announced a new mission to Mars with astronauts launching rocket orbiting the moon.",
    "Hockey / Sports": "The team scored a goal in overtime winning the championship hockey league playoffs.",
    "Religion": "The church discussed spiritual faith prayer belief and religious doctrine on Sunday.",
    "Computers / Graphics": "The GPU renders 3D images using OpenGL shaders and texture mapping algorithms.",
    "Cars / Auto": "The new car model features a turbocharged engine with fuel injection and ABS brakes.",
    "Medicine": "The patient was treated with antibiotics and surgery to remove the infected tissue.",
    "Politics / Guns": "Congress debated new gun legislation on the second amendment and firearms regulations.",
}
example_choice = st.selectbox("💡 Try an example", ["(type your own)"] + list(example_texts.keys()))
default_text = "" if example_choice == "(type your own)" else example_texts[example_choice]

user_text = st.text_area(
    "Type or paste news article text below:",
    value=default_text,
    height=160,
    placeholder="Paste or type a news article excerpt here..."
)
st.markdown("</div>", unsafe_allow_html=True)

if st.button("📰  Classify Article"):
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        proba = pipe.predict_proba([user_text])[0]
        pred_idx = np.argmax(proba)
        pred_cat = target_names[pred_idx]
        confidence = proba[pred_idx] * 100

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.85rem; color:rgba(200,150,255,0.6); text-transform:uppercase; letter-spacing:1px;">Predicted Category</div>
            <div class="big-cat">📂 {pred_cat}</div>
            <div style="font-size:0.9rem; color:rgba(220,180,255,0.75);">
                Confidence: <strong style="color:#c084fc;">{confidence:.1f}%</strong>
            </div>
            <div class="conf-bar"><div class="conf-fill" style="width:{confidence:.1f}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass" style="margin-top:18px;">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📊 All Category Probabilities</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Category": target_names,
            "Probability (%)": [round(p * 100, 2) for p in proba]
        }).sort_values("Probability (%)", ascending=False).set_index("Category")
        st.bar_chart(prob_df, color="#c084fc")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>📰 20 Newsgroups Dataset · Multinomial Naive Bayes · TF-IDF · Streamlit</div>", unsafe_allow_html=True)
