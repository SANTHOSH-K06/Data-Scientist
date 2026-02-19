import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ---- Page Config ----
st.set_page_config(
    page_title="Iris Classifier | Decision Tree",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #000a1a 0%, #00111f 50%, #000810 100%);
    color: #e0f8ff;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, rgba(0,180,220,0.35), rgba(0,140,200,0.25));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 40px rgba(0,180,220,0.2);
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px !important;
}
.hero p { color: rgba(150,220,255,0.75); font-size: 0.95rem; margin: 0; }

.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(56,189,248,0.14);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}
.sec-title {
    font-size: 0.9rem; font-weight: 700; color: #38bdf8;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0c4a6e, #0284c7) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85em 2em !important;
    box-shadow: 0 6px 20px rgba(2,132,199,0.45) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.stSlider label, .stSelectbox label { color: #93c5fd !important; font-weight: 600 !important; }

.result-card {
    border-radius: 16px;
    padding: 26px 22px;
    margin-top: 20px;
    animation: fadeIn 0.5s ease;
}
.result-setosa     { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); border-left: 5px solid #38bdf8; }
.result-versicolor { background: rgba(129,140,248,0.12); border: 1px solid rgba(129,140,248,0.3); border-left: 5px solid #818cf8; }
.result-virginica  { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3); border-left: 5px solid #34d399; }

.big-class { font-size: 2rem; font-weight: 800; margin: 8px 0; }

.conf-bar { background: rgba(255,255,255,0.08); border-radius: 50px; height: 10px; margin-top: 12px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 50px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000810, #000510) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
}
[data-testid="stSidebar"] * { color: #93c5fd !important; }
.sidebar-card {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 12px;
    padding: 13px;
    font-size: 0.84rem;
    line-height: 1.65;
    margin-bottom: 12px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer { text-align: center; padding: 16px; color: rgba(100,200,255,0.3); font-size: 0.78rem; margin-top: 30px; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🌸 About")
    st.markdown("""<div class="sidebar-card">
    Classifies Iris flowers into 3 species using a <strong>Decision Tree</strong>. 
    Tune depth and criterion to see their effect on accuracy.
    </div>""", unsafe_allow_html=True)
    max_depth = st.slider("Max Tree Depth", 1, 15, 4)
    criterion = st.selectbox("Criterion", ["gini", "entropy"])
    st.markdown("---")
    st.markdown("### 🌺 Classes")
    st.markdown("""<div class="sidebar-card">
    🔵 <b>Setosa</b> — Small petals, distinct<br>
    🟣 <b>Versicolor</b> — Medium, mixed<br>
    🟢 <b>Virginica</b> — Large petals
    </div>""", unsafe_allow_html=True)

# ---- Train ----
@st.cache_data
def train(max_depth, criterion):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    cv = cross_val_score(clf, X, y, cv=5).mean()
    return clf, acc, cv

clf, acc, cv = train(max_depth, criterion)
iris = load_iris()

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>🌸 Iris Flower Classifier</h1>
    <p>Classify Iris flowers into 3 species using a Decision Tree</p>
</div>
""", unsafe_allow_html=True)

# ---- Metrics ----
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="glass"><div class="sec-title">🎯 Accuracy</div><div style="font-size:1.8rem;font-weight:800;color:#38bdf8;">{acc*100:.2f}%</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="glass"><div class="sec-title">📊 CV Score (5-fold)</div><div style="font-size:1.8rem;font-weight:800;color:#38bdf8;">{cv*100:.2f}%</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="glass"><div class="sec-title">🌲 Tree Depth</div><div style="font-size:1.8rem;font-weight:800;color:#38bdf8;">{max_depth}</div></div>', unsafe_allow_html=True)

# ---- Inputs ----
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🧪 Flower Measurements</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    sepal_len = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_wid = st.slider("Sepal Width (cm)",  1.5, 5.0, 3.0, 0.1)
with c2:
    petal_len = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
    petal_wid = st.slider("Petal Width (cm)",  0.1, 2.6, 1.2, 0.1)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("🌸  Classify Iris Flower"):
    x = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0]
    conf = proba[pred] * 100

    class_names = iris.target_names
    class_name  = class_names[pred].capitalize()
    colors = ["#38bdf8","#818cf8","#34d399"]
    css_cls = ["result-setosa","result-versicolor","result-virginica"]
    emojis  = ["🔵","🟣","🟢"]

    st.markdown(f"""
    <div class="result-card {css_cls[pred]}">
        <div class="big-class" style="color:{colors[pred]};">{emojis[pred]} Iris {class_name}</div>
        <div style="font-size:0.9rem; color:rgba(180,220,255,0.75);">
            Sepal: {sepal_len}×{sepal_wid} cm &nbsp;·&nbsp; Petal: {petal_len}×{petal_wid} cm
        </div>
        <div style="margin-top:12px; font-size:0.85rem; color:rgba(150,200,255,0.6);">
            Confidence: <strong style="color:{colors[pred]};">{conf:.1f}%</strong>
        </div>
        <div class="conf-bar">
            <div class="conf-fill" style="width:{conf:.1f}%; background:{colors[pred]};"></div>
        </div>
        <div style="font-size:0.78rem; color:rgba(150,200,255,0.5); margin-top:6px;">
            Setosa: {proba[0]*100:.1f}% &nbsp;|&nbsp; Versicolor: {proba[1]*100:.1f}% &nbsp;|&nbsp; Virginica: {proba[2]*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Feature importance ----
st.markdown('<div class="glass" style="margin-top:20px;">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">⭐ Feature Importance</div>', unsafe_allow_html=True)
imp_df = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": clf.feature_importances_
}).sort_values("Importance", ascending=False).set_index("Feature")
st.bar_chart(imp_df, color="#38bdf8")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>🌸 Iris Dataset · Decision Tree · Streamlit</div>", unsafe_allow_html=True)
