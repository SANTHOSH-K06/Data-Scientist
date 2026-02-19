import streamlit as st
import numpy as np
import joblib
import os

# ---------------- Load Model ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_wine.pkl")
model = joblib.load(MODEL_PATH)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Wine Quality Predictor | AI-Powered",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Premium Custom CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0d0014 0%, #1a0008 40%, #0d001a 100%);
        color: #f5f0ff;
        min-height: 100vh;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(139,30,45,0.6), rgba(90,0,80,0.6));
        border: 1px solid rgba(255,100,120,0.25);
        border-radius: 20px;
        padding: 36px 32px 28px;
        text-align: center;
        margin-bottom: 30px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 40px rgba(139,30,45,0.3), 0 0 0 1px rgba(255,255,255,0.05);
    }

    .hero-banner h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff8fa0, #e040fb, #f48fb1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 6px !important;
    }

    .hero-banner p {
        color: rgba(240,220,255,0.8);
        font-size: 1rem;
        margin: 0;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e040fb;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 18px;
    }

    /* Number inputs */
    .stNumberInput label {
        color: #d4b8f0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    .stNumberInput input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(224,64,251,0.35) !important;
        border-radius: 10px !important;
        color: #fff !important;
        font-size: 1rem !important;
        padding: 8px 12px !important;
        transition: all 0.3s ease !important;
    }

    .stNumberInput input:focus {
        border-color: rgba(224,64,251,0.8) !important;
        box-shadow: 0 0 0 3px rgba(224,64,251,0.15) !important;
        outline: none !important;
    }

    /* Slider */
    .stSlider label {
        color: #d4b8f0 !important;
        font-weight: 600 !important;
    }

    /* Predict Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #8b1e2d 0%, #b026b0 50%, #8b1e2d 100%) !important;
        background-size: 200% 200% !important;
        color: white !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85em 2em !important;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.4s ease !important;
        box-shadow: 0 6px 20px rgba(176,38,176,0.4) !important;
        margin-top: 8px !important;
    }

    .stButton > button:hover {
        background-position: right center !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(176,38,176,0.55) !important;
    }

    /* Prediction Result Card */
    .result-high {
        background: linear-gradient(135deg, rgba(0,200,100,0.12), rgba(0,255,130,0.06));
        border: 1px solid rgba(0,255,130,0.25);
        border-left: 5px solid #00e676;
        border-radius: 16px;
        padding: 28px 24px;
        margin-top: 24px;
        animation: fadeIn 0.6s ease;
    }

    .result-low {
        background: linear-gradient(135deg, rgba(255,80,80,0.12), rgba(255,40,40,0.06));
        border: 1px solid rgba(255,80,80,0.25);
        border-left: 5px solid #ff5252;
        border-radius: 16px;
        padding: 28px 24px;
        margin-top: 24px;
        animation: fadeIn 0.6s ease;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .result-msg {
        font-size: 0.95rem;
        color: rgba(240,230,255,0.85);
        line-height: 1.6;
    }

    /* Confidence Bar */
    .confidence-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        height: 10px;
        margin-top: 14px;
        overflow: hidden;
    }

    .confidence-fill-high {
        height: 100%;
        background: linear-gradient(90deg, #00c853, #00e676);
        border-radius: 50px;
        animation: grow 0.8s ease forwards;
    }

    .confidence-fill-low {
        height: 100%;
        background: linear-gradient(90deg, #ff5252, #ff1744);
        border-radius: 50px;
        animation: grow 0.8s ease forwards;
    }

    /* Metric boxes */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
    }

    .metric-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }

    .metric-label {
        font-size: 0.75rem;
        color: rgba(200,180,255,0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .metric-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e040fb;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #120020 0%, #0d0018 100%) !important;
        border-right: 1px solid rgba(224,64,251,0.15) !important;
    }

    [data-testid="stSidebar"] * {
        color: #d4b8f0 !important;
    }

    .sidebar-info {
        background: rgba(224,64,251,0.08);
        border: 1px solid rgba(224,64,251,0.2);
        border-radius: 12px;
        padding: 14px;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08) !important;
        margin: 20px 0 !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        margin-top: 36px;
        padding: 16px;
        color: rgba(200,180,255,0.5);
        font-size: 0.78rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes grow {
        from { width: 0%; }
    }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🍷 About This App")
    st.markdown("""
    <div class="sidebar-info">
    This app uses a <strong>Random Forest Classifier</strong> trained on wine chemical properties to predict quality.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class="sidebar-info">
    <b>Algorithm:</b> Random Forest<br>
    <b>Task:</b> Binary Classification<br>
    <b>Classes:</b> High Quality / Low Quality<br>
    <b>Features Used:</b> 4 key chemical properties
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 Chemical Properties")
    st.markdown("""
    <div class="sidebar-info">
    • <b>Alcohol:</b> % volume content<br>
    • <b>Sulphates:</b> Preservative agent<br>
    • <b>Volatile Acidity:</b> Acetic acid level<br>
    • <b>pH:</b> Acidity/Alkalinity scale
    </div>
    """, unsafe_allow_html=True)

# ---- Hero Banner ----
st.markdown("""
<div class="hero-banner">
    <h1>🍷 Wine Quality Predictor</h1>
    <p>Enter chemical properties below to predict wine quality using AI</p>
</div>
""", unsafe_allow_html=True)

# ---- Input Section ----
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Chemical Properties</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input(
        "Alcohol (%)",
        min_value=0.0, max_value=20.0, value=10.0, step=0.1,
        help="Alcohol content by volume (%)"
    )
    volatile_acidity = st.number_input(
        "Volatile Acidity (g/L)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.01,
        help="Acetic acid level — high values give an unpleasant vinegar taste"
    )

with col2:
    sulphates = st.number_input(
        "Sulphates (g/L)",
        min_value=0.0, max_value=2.0, value=0.65, step=0.01,
        help="Wine additive contributing to SO2 levels as an antimicrobial agent"
    )
    pH = st.number_input(
        "pH Level",
        min_value=2.0, max_value=5.0, value=3.3, step=0.01,
        help="Describes how acidic (2–3) or alkaline (5+) the wine is"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---- Quick Reference ----
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Input Summary</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">Alcohol</div>
        <div class="metric-val">{alcohol:.1f}%</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Sulphates</div>
        <div class="metric-val">{sulphates:.2f} g/L</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Volatile Acidity</div>
        <div class="metric-val">{volatile_acidity:.2f} g/L</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">pH Level</div>
        <div class="metric-val">{pH:.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Predict Button ----
predict_clicked = st.button("🔍  Predict Wine Quality", use_container_width=True)

# ---- Prediction Result ----
if predict_clicked:
    input_data = np.array([[alcohol, sulphates, volatile_acidity, pH]])
    prediction = model.predict(input_data)[0]

    # Try to get probability if available
    try:
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
    except Exception:
        confidence = 80.0

    if prediction == 1:
        quality = "High Quality Wine"
        message = (
            "This wine demonstrates excellent chemical balance. "
            "The combination of its alcohol content, low volatile acidity, and "
            "optimal sulphate-to-pH ratio suggests a pleasant, well-rounded flavor profile."
        )
        emoji = "🍷"
        css_class = "result-high"
        bar_class = "confidence-fill-high"
        color = "#00e676"
    else:
        quality = "Low Quality Wine"
        message = (
            "The chemical profile of this wine suggests some imbalance. "
            "High volatile acidity or suboptimal alcohol-to-sulphate ratios may "
            "result in a less pleasant taste and aroma experience."
        )
        emoji = "⚠️"
        css_class = "result-low"
        bar_class = "confidence-fill-low"
        color = "#ff5252"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title" style="color:{color};">{emoji} {quality}</div>
        <div class="result-msg">{message}</div>
        <div style="margin-top:16px; font-size:0.85rem; color:rgba(200,180,255,0.7);">
            Model Confidence
        </div>
        <div class="confidence-bar">
            <div class="{bar_class}" style="width:{confidence:.1f}%;"></div>
        </div>
        <div style="font-size:0.8rem; color:rgba(200,180,255,0.6); margin-top:4px; text-align:right;">
            {confidence:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class="app-footer">
    🤖 Powered by Random Forest Classification &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; 
    Data Science Portfolio
</div>
""", unsafe_allow_html=True)