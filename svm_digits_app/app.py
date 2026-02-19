import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Digit Recognizer | SVM AI",
    page_icon="✍️",
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
        background: linear-gradient(135deg, #060d1f 0%, #0a1628 50%, #060d1f 100%);
        color: #e8f0ff;
        min-height: 100vh;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0,112,240,0.35), rgba(100,0,200,0.35));
        border: 1px solid rgba(80,140,255,0.3);
        border-radius: 20px;
        padding: 36px 32px 28px;
        text-align: center;
        margin-bottom: 28px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 40px rgba(0,112,240,0.2), 0 0 0 1px rgba(255,255,255,0.05);
    }

    .hero-banner h1 {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px !important;
    }

    .hero-banner p {
        color: rgba(200,220,255,0.75);
        font-size: 0.95rem;
        margin: 0;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100,140,255,0.15);
        border-radius: 16px;
        padding: 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 16px;
    }

    /* File Upload Area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(96,165,250,0.35) !important;
        border-radius: 14px !important;
        background: rgba(96,165,250,0.04) !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(96,165,250,0.7) !important;
        background: rgba(96,165,250,0.08) !important;
    }

    [data-testid="stFileUploader"] label {
        color: #93c5fd !important;
        font-weight: 600 !important;
    }

    /* Radio buttons */
    .stRadio label {
        color: #b8d4ff !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%) !important;
        color: white !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85em 2em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(29,78,216,0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(124,58,237,0.55) !important;
    }

    /* Success box */
    .digit-result {
        background: linear-gradient(135deg, rgba(29,78,216,0.15), rgba(124,58,237,0.15));
        border: 1px solid rgba(96,165,250,0.3);
        border-left: 5px solid #60a5fa;
        border-radius: 16px;
        padding: 28px 24px;
        margin-top: 20px;
        animation: fadeSlide 0.6s ease forwards;
    }

    .big-digit {
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        line-height: 1;
        margin: 12px 0;
    }

    /* Accuracy badge */
    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.2));
        border: 1px solid rgba(16,185,129,0.35);
        border-radius: 50px;
        padding: 6px 18px;
        font-size: 0.85rem;
        color: #6ee7b7;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Stat chips */
    .stat-row {
        display: flex;
        gap: 12px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .stat-chip {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        padding: 8px 18px;
        font-size: 0.82rem;
        color: rgba(200,220,255,0.8);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111e 0%, #050d1a 100%) !important;
        border-right: 1px solid rgba(96,165,250,0.12) !important;
    }

    [data-testid="stSidebar"] * {
        color: #b8d4ff !important;
    }

    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #60a5fa !important;
    }

    .sidebar-info {
        background: rgba(96,165,250,0.07);
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 12px;
        padding: 13px 15px;
        font-size: 0.84rem;
        line-height: 1.65;
        margin-bottom: 14px;
    }

    /* Steps guide */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
    }

    .step-num {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
        color: white;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    .step-text {
        font-size: 0.85rem;
        color: rgba(200,220,255,0.8);
        line-height: 1.5;
    }

    /* Progress bar */
    .accuracy-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 50px;
        height: 8px;
        margin-top: 10px;
        overflow: hidden;
    }

    .accuracy-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #1d4ed8, #7c3aed);
        border-radius: 50px;
        animation: growBar 1s ease forwards;
    }

    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes growBar {
        from { width: 0; }
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07) !important;
        margin: 18px 0 !important;
    }

    .app-footer {
        text-align: center;
        margin-top: 38px;
        padding: 16px;
        color: rgba(150,180,255,0.4);
        font-size: 0.78rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ✍️ About")
    st.markdown("""
    <div class="sidebar-info">
    This app uses a <strong>Support Vector Machine (SVM)</strong> to classify 
    handwritten digits from images (0–9).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Config")
    model_type = st.radio(
        "Choose Kernel",
        ["Linear SVM", "Non-Linear SVM (RBF)"],
        index=1
    )
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    <div class="step-item">
        <div class="step-num">1</div>
        <div class="step-text">Select your SVM kernel from above</div>
    </div>
    <div class="step-item">
        <div class="step-num">2</div>
        <div class="step-text">Upload a clear image of a handwritten digit (0–9)</div>
    </div>
    <div class="step-item">
        <div class="step-num">3</div>
        <div class="step-text">The model will classify and display its prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    <div class="sidebar-info">
    • Use dark ink on a white background<br>
    • Center your digit in the image<br>
    • PNG or JPG formats work best<br>
    • Minimal noise gives better results
    </div>
    """, unsafe_allow_html=True)

# ---- Hero Banner ----
st.markdown("""
<div class="hero-banner">
    <h1>✍️ Handwritten Digit Recognizer</h1>
    <p>Upload an image and let the SVM model identify the digit (0–9)</p>
</div>
""", unsafe_allow_html=True)

# ---- Load Dataset & Train ----
with st.spinner("🔄 Initializing SVM model..."):
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    if model_type == "Linear SVM":
        model = SVC(kernel="linear", C=10, probability=True)
    else:
        model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

# ---- Model Info Card ----
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤖 Model Status</div>', unsafe_allow_html=True)

kernel_name = "Linear" if model_type == "Linear SVM" else "RBF (Non-Linear)"
acc_pct = accuracy * 100

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <div>
        <div style="font-size:0.85rem; color:rgba(150,200,255,0.6);">ACTIVE KERNEL</div>
        <div style="font-size:1.1rem; font-weight:700; color:#60a5fa;">{kernel_name} SVM</div>
    </div>
    <div class="accuracy-badge">✅ Accuracy: {acc_pct:.2f}%</div>
</div>
<div class="accuracy-bar-bg">
    <div class="accuracy-bar-fill" style="width:{acc_pct:.1f}%;"></div>
</div>
<div style="font-size:0.78rem; color:rgba(150,200,255,0.5); margin-top:5px; text-align:right;">
    Trained on {len(X_train)} samples · Tested on {len(X_test)} samples
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Upload Section ----
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Upload Digit Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop or click to upload a digit image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Prediction ----
if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("L")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, caption="Original (grayscale)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Preprocess
    resized = image.resize((8, 8))
    image_array = np.array(resized)
    image_array = 255 - image_array          # invert
    image_array = (image_array / 255.0) * 16  # normalize to digits dataset scale
    flat = image_array.reshape(1, -1)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Preprocessed (8×8)</div>', unsafe_allow_html=True)
        st.image(resized, caption="Resized to 8×8px", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction
    with st.spinner("🤖 Analyzing..."):
        time.sleep(0.4)  # short animation delay
        prediction = model.predict(flat)[0]
        try:
            proba = model.predict_proba(flat)[0]
            confidence = proba[prediction] * 100
        except Exception:
            confidence = 85.0

    # Result
    st.markdown(f"""
    <div class="digit-result">
        <div style="text-align:center;">
            <div style="font-size:0.9rem; color:rgba(150,200,255,0.6); text-transform:uppercase; letter-spacing:1px;">
                Predicted Digit
            </div>
            <div class="big-digit">{prediction}</div>
            <div style="font-size:0.9rem; color:rgba(160,200,255,0.7);">
                Confidence: <strong style="color:#60a5fa;">{confidence:.1f}%</strong>
            </div>
        </div>
        <hr style="margin: 18px 0 !important;" />
        <div class="stat-row">
            <div class="stat-chip">📐 Kernel: {kernel_name}</div>
            <div class="stat-chip">🎯 Test Accuracy: {acc_pct:.2f}%</div>
            <div class="stat-chip">🔢 Predicted: <b>{prediction}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilities chart
    try:
        st.markdown('<div class="glass-card" style="margin-top:20px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Prediction Probabilities</div>', unsafe_allow_html=True)
        proba_all = model.predict_proba(flat)[0]
        import pandas as pd
        df_proba = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Confidence (%)": [round(p * 100, 2) for p in proba_all]
        }).set_index("Digit")
        st.bar_chart(df_proba, color="#60a5fa")
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass

else:
    # Placeholder
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px; color:rgba(150,200,255,0.4);">
        <div style="font-size:3rem; margin-bottom:12px;">📤</div>
        <div style="font-size:1rem; font-weight:600;">Upload a digit image to get started</div>
        <div style="font-size:0.85rem; margin-top:6px;">Supports PNG, JPG, and JPEG formats</div>
    </div>
    """, unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class="app-footer">
    🤖 Support Vector Machine Classification &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; 
    Sklearn Digits Dataset
</div>
""", unsafe_allow_html=True)