import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IntelliFeature | Feature Selection Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Premium Custom CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --primary: #00f2fe;
        --secondary: #4facfe;
        --accent: #f093fb;
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Card */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
    }

    .metric-card {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border-left: 4px solid var(--primary);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(to right, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    .prediction-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 242, 254, 0.3);
    }

    /* Gradient Button */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data & Models
# -----------------------------
@st.cache_resource
def load_assets():
    data = load_breast_cancer()
    scaler = joblib.load("scaler.pkl")
    baseline_model = joblib.load("baseline_model.pkl")
    filter_model = joblib.load("filter_model.pkl")
    wrapper_model = joblib.load("wrapper_model.pkl")
    filter_selector = joblib.load("filter_selector.pkl")
    wrapper_selector = joblib.load("wrapper_selector.pkl")
    return {
        "data": data,
        "scaler": scaler,
        "baseline": baseline_model,
        "filter": filter_model,
        "wrapper": wrapper_model,
        "f_selector": filter_selector,
        "w_selector": wrapper_selector
    }

assets = load_assets()
data = assets["data"]
feature_names = data.feature_names
X = data.data
y = data.target

# -----------------------------
# Header
# -----------------------------
st.title("🎯 IntelliFeature")
st.markdown("#### Precision Feature Selection Engine for Healthcare Diagnostics")
st.write("Optimizing Breast Cancer classification through advanced Filter and Wrapper methodologies.")

st.divider()

# -----------------------------
# Metrics Section
# -----------------------------
X_scaled = assets["scaler"].transform(X)

b_acc = accuracy_score(y, assets["baseline"].predict(X_scaled))
f_acc = accuracy_score(y, assets["filter"].predict(assets["f_selector"].transform(X_scaled)))
w_acc = accuracy_score(y, assets["wrapper"].predict(assets["w_selector"].transform(X_scaled)))

m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline (Full)</div>
        <div class="metric-value">{b_acc:.2%}</div>
        <div style="color:#94a3b8; font-size: 0.8rem;">30 Features</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #f093fb;">
        <div class="metric-label">Filter (SelectKBest)</div>
        <div class="metric-value">{f_acc:.2%}</div>
        <div style="color:#94a3b8; font-size: 0.8rem;">10 Features</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #4facfe;">
        <div class="metric-label">Wrapper (RFE)</div>
        <div class="metric-value">{w_acc:.2%}</div>
        <div style="color:#94a3b8; font-size: 0.8rem;">10 Features</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------
# Comparisons & Selection Sidebars
# -----------------------------
c1, c2 = st.columns([1, 1.2])

with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🛠️ Selected Features Comparison")
    
    filter_feats = feature_names[assets["f_selector"].get_support()]
    wrapper_feats = feature_names[assets["w_selector"].get_support()]
    
    tab1, tab2 = st.tabs(["Filter Method", "Wrapper Method"])
    
    with tab1:
        st.info("Statistical significance-based selection (ANOVA F-statistic).")
        for f in filter_feats:
            st.markdown(f"- `{f}`")
            
    with tab2:
        st.info("Recursive elimination based on model weights.")
        for f in wrapper_feats:
            st.markdown(f"- `{f}`")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Visualization")
    
    # Simple Plotly Chart for Accuracy
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Baseline", "Filter", "Wrapper"],
        y=[b_acc, f_acc, w_acc],
        marker_color=["#00f2fe", "#f093fb", "#4facfe"],
        text=[f"{b_acc:.2%}", f"{f_acc:.2%}", f"{w_acc:.2%}"],
        textposition='auto',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=350,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Interactive Prediction
# -----------------------------
st.divider()
st.subheader("🔍 Real-time Diagnostics")

with st.expander("Diagnostic Input Panel", expanded=True):
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        method = st.radio("Intelligence Engine", ["Baseline", "Filter Method", "Wrapper Method"], horizontal=True)
    
    with p_col2:
        st.write("Adjust clinical parameters below for prediction:")

    inputs = []
    # Using 10 features for demo as per original repo logic
    i_cols = st.columns(5)
    for i in range(10):
        with i_cols[i % 5]:
            val = st.number_input(feature_names[i].capitalize(), value=float(X[0,i]), help=f"Value for {feature_names[i]}")
            inputs.append(val)

    if st.button("RUN DIAGNOSIS"):
        input_array = np.array([inputs])
        # Padding logic same as original but cleaned up
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = assets["scaler"].transform(padded)
        
        if method == "Baseline":
            pred = assets["baseline"].predict(scaled)
            prob = assets["baseline"].predict_proba(scaled)[0][pred[0]]
        elif method == "Filter Method":
            selected = assets["f_selector"].transform(scaled)
            pred = assets["filter"].predict(selected)
            prob = assets["filter"].predict_proba(selected)[0][pred[0]]
        else:
            selected = assets["w_selector"].transform(scaled)
            pred = assets["wrapper"].predict(selected)
            prob = assets["wrapper"].predict_proba(selected)[0][pred[0]]

        st.write("")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if pred[0] == 1:
                st.success("### ✅ BENIGN")
            else:
                st.error("### ⚠️ MALIGNANT")
        
        with res_col2:
            st.write(f"**Confidence Level:** {prob:.2%}")
            st.progress(float(prob))

st.caption("Disclaimer: This tool is for demonstration of feature selection techniques only and should not be used for medical diagnosis.")
