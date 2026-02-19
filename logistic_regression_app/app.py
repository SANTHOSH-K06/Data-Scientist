import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ---- Page Config ----
st.set_page_config(
    page_title="Cancer Classifier | Logistic Regression",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #000f0a 0%, #001a10 50%, #000c08 100%);
    color: #e8fff4;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, rgba(0,180,100,0.35), rgba(0,120,80,0.3));
    border: 1px solid rgba(0,220,130,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 40px rgba(0,180,100,0.2);
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #4ade80, #22d3ee, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px !important;
}
.hero p { color: rgba(150,255,200,0.7); font-size: 0.95rem; margin: 0; }

.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,220,130,0.14);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}
.sec-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #34d399;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #065f46, #059669) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85em 2em !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.result-benign {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-left: 5px solid #34d399;
    border-radius: 16px;
    padding: 26px;
    margin-top: 20px;
    animation: fadeIn 0.5s ease;
}
.result-malignant {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.3);
    border-left: 5px solid #f87171;
    border-radius: 16px;
    padding: 26px;
    margin-top: 20px;
    animation: fadeIn 0.5s ease;
}
.result-big {
    font-size: 2rem;
    font-weight: 800;
    margin: 10px 0;
}

.conf-bar { background: rgba(255,255,255,0.08); border-radius: 50px; height: 10px; margin-top: 12px; overflow: hidden; }
.conf-fill-g { height: 100%; background: linear-gradient(90deg, #059669, #34d399); border-radius: 50px; }
.conf-fill-r { height: 100%; background: linear-gradient(90deg, #dc2626, #f87171); border-radius: 50px; }

.stSlider label, .stNumberInput label { color: #86efac !important; font-weight: 600 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000e08, #000a06) !important;
    border-right: 1px solid rgba(52,211,153,0.12) !important;
}
[data-testid="stSidebar"] * { color: #86efac !important; }

.sidebar-card {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.18);
    border-radius: 12px;
    padding: 13px;
    font-size: 0.84rem;
    line-height: 1.65;
    margin-bottom: 12px;
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer { text-align: center; padding: 16px; color: rgba(100,255,180,0.3); font-size: 0.78rem; margin-top: 30px; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🔬 About")
    st.markdown("""<div class="sidebar-card">
    Classifies tumors as <strong>Benign</strong> or <strong>Malignant</strong> using Logistic Regression 
    on the Wisconsin Breast Cancer dataset.
    </div>""", unsafe_allow_html=True)
    st.markdown("### ⚙️ Parameters")
    C_val = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01)
    max_iter = st.slider("Max Iterations", 100, 1000, 200, 50)
    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.markdown("""<div class="sidebar-card">
    <b>Dataset:</b> Breast Cancer Wisconsin<br>
    <b>Samples:</b> 569<br>
    <b>Features:</b> 30 numeric<br>
    <b>Classes:</b> Benign (357) / Malignant (212)
    </div>""", unsafe_allow_html=True)

# ---- Load & Train ----
@st.cache_data
def train_model(C, max_iter):
    data = load_breast_cancer()
    X, y = data.data, data.target
    feat_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    clf.fit(X_train_s, y_train)
    acc = clf.score(X_test_s, y_test)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    return clf, scaler, acc, auc, feat_names, data.feature_names[:10]

clf, scaler, acc, auc, feat_names, top_feats = train_model(C_val, max_iter)

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>🔬 Tumor Classifier</h1>
    <p>Detect Benign vs Malignant tumors using Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ---- Metrics ----
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="glass"><div class="sec-title">✅ Accuracy</div><div style="font-size:1.8rem;font-weight:800;color:#34d399;">{acc*100:.2f}%</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="glass"><div class="sec-title">📊 ROC-AUC</div><div style="font-size:1.8rem;font-weight:800;color:#34d399;">{auc:.4f}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="glass"><div class="sec-title">🔧 C Parameter</div><div style="font-size:1.8rem;font-weight:800;color:#34d399;">{C_val}</div></div>', unsafe_allow_html=True)

# ---- Inputs ----
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🧪 Tumor Features (Top 10)</div>', unsafe_allow_html=True)
st.info("Adjust the 10 key features below — defaults are dataset mean values.")

data = load_breast_cancer()
means = data.data.mean(axis=0)

cols_a, cols_b = st.columns(2)
user_vals = []
for i, fname in enumerate(data.feature_names[:10]):
    col = cols_a if i % 2 == 0 else cols_b
    with col:
        val = st.number_input(
            fname,
            value=float(round(means[i], 4)),
            format="%.4f",
            step=float(round(means[i] * 0.05, 5))
        )
        user_vals.append(val)

# Fill remaining 20 features with means
full_input = list(user_vals) + list(means[10:])
st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔬  Classify Tumor"):
    raw = np.array([full_input])
    scaled = scaler.transform(raw)
    pred = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    confidence = proba[pred] * 100

    if pred == 1:
        label, css, fill_cls, color, emoji = "Benign", "result-benign", "conf-fill-g", "#34d399", "✅"
    else:
        label, css, fill_cls, color, emoji = "Malignant", "result-malignant", "conf-fill-r", "#f87171", "⚠️"

    st.markdown(f"""
    <div class="{css}">
        <div class="result-big" style="color:{color};">{emoji} {label} Tumor</div>
        <div style="font-size:0.9rem; color:rgba(200,255,220,0.7);">
            The model predicts this tumor is <strong>{label}</strong> based on the provided features.
        </div>
        <div style="margin-top:14px; font-size:0.85rem; color:rgba(180,255,210,0.6);">
            Confidence: <strong style="color:{color};">{confidence:.1f}%</strong>
        </div>
        <div class="conf-bar"><div class="{fill_cls}" style="width:{confidence:.1f}%;"></div></div>
        <div style="font-size:0.75rem; color:rgba(180,255,210,0.5); margin-top:4px;">
            Benign: {proba[1]*100:.1f}% &nbsp;|&nbsp; Malignant: {proba[0]*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>🔬 Breast Cancer Wisconsin Dataset · Logistic Regression · Streamlit</div>", unsafe_allow_html=True)
