import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(
    page_title="House Price Predictor | Linear Regression",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0f0800 0%, #1a1000 50%, #0c0800 100%);
    color: #fff8ee;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, rgba(220,100,0,0.4), rgba(180,60,0,0.3));
    border: 1px solid rgba(255,140,0,0.25);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 40px rgba(220,100,0,0.2);
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ffb347, #ff6b00, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px !important;
}
.hero p { color: rgba(255,220,160,0.8); font-size: 0.95rem; margin: 0; }

.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,140,0,0.15);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
}
.sec-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #ff9500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
}
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.metric-card {
    flex: 1; min-width: 120px;
    background: rgba(255,149,0,0.08);
    border: 1px solid rgba(255,149,0,0.2);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.metric-label { font-size: 0.72rem; color: rgba(255,200,100,0.6); text-transform: uppercase; letter-spacing: 1px; }
.metric-val { font-size: 1.3rem; font-weight: 800; color: #ffb347; margin-top: 4px; }

.result-box {
    background: linear-gradient(135deg, rgba(255,149,0,0.12), rgba(255,100,0,0.08));
    border: 1px solid rgba(255,149,0,0.3);
    border-left: 5px solid #ff9500;
    border-radius: 16px;
    padding: 26px 22px;
    margin-top: 20px;
    animation: fadeIn 0.5s ease;
}
.big-price {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffb347, #ff6b00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c05000, #ff6b00) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85em 2em !important;
    box-shadow: 0 6px 20px rgba(255,106,0,0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.stSlider label, .stSelectbox label { color: #ffd080 !important; font-weight: 600 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0800, #080600) !important;
    border-right: 1px solid rgba(255,149,0,0.12) !important;
}
[data-testid="stSidebar"] * { color: #ffd080 !important; }

.sidebar-card {
    background: rgba(255,149,0,0.07);
    border: 1px solid rgba(255,149,0,0.18);
    border-radius: 12px;
    padding: 13px;
    font-size: 0.84rem;
    line-height: 1.6;
    margin-bottom: 12px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.footer { text-align: center; padding: 16px; color: rgba(255,200,100,0.35); font-size: 0.78rem; margin-top: 30px; border-top: 1px solid rgba(255,255,255,0.05); }
@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🏠 About")
    st.markdown("""<div class="sidebar-card">
    Predicts <strong>California house prices</strong> using Linear, Lasso, and Ridge regression on 8 features.
    </div>""", unsafe_allow_html=True)

    model_choice = st.selectbox("📐 Regression Model", ["Linear Regression", "Lasso Regression", "Ridge Regression"])
    alpha_val = 1.0
    if model_choice in ["Lasso Regression", "Ridge Regression"]:
        alpha_val = st.slider("Regularization (α)", 0.01, 10.0, 1.0, 0.01)

    st.markdown("---")
    st.markdown("### 📋 Features Used")
    st.markdown("""<div class="sidebar-card">
    • MedInc — Median income<br>
    • HouseAge — House age (years)<br>
    • AveRooms — Avg rooms<br>
    • AveBedrms — Avg bedrooms<br>
    • Population — Block population<br>
    • AveOccup — Avg occupants<br>
    • Latitude / Longitude
    </div>""", unsafe_allow_html=True)

# ---- Load & Train ----
@st.cache_data
def load_and_train(model_name, alpha):
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Lasso Regression":
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, scaler, r2, rmse, df, y_test, y_pred, housing.feature_names

model, scaler, r2, rmse, df, y_test, y_pred, feat_names = load_and_train(model_choice, alpha_val)

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Predictor</h1>
    <p>California housing price prediction powered by Linear Regression models</p>
</div>
""", unsafe_allow_html=True)

# ---- Model Performance ----
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="glass">
    <div class="sec-title">📊 Model</div>
    <div style="font-weight:700; color:#ffb347; font-size:1rem;">{model_choice}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="glass">
    <div class="sec-title">🎯 R² Score</div>
    <div style="font-weight:800; color:#ffb347; font-size:1.6rem;">{r2:.4f}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="glass">
    <div class="sec-title">📉 RMSE</div>
    <div style="font-weight:800; color:#ffb347; font-size:1.6rem;">${rmse*100000:.0f}</div>
    </div>""", unsafe_allow_html=True)

# ---- Input Section ----
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🔧 Input Features</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    med_inc    = st.slider("Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    house_age  = st.slider("House Age (years)", 1, 52, 20)
    ave_rooms  = st.slider("Avg Rooms", 1.0, 15.0, 6.0, 0.1)
    ave_bedrms = st.slider("Avg Bedrooms", 0.5, 5.0, 1.1, 0.1)
with c2:
    population = st.slider("Block Population", 3, 4000, 1200)
    ave_occup  = st.slider("Avg Occupants", 1.0, 6.0, 3.0, 0.1)
    latitude   = st.slider("Latitude", 32.5, 42.0, 36.0, 0.1)
    longitude  = st.slider("Longitude", -124.5, -114.0, -120.0, 0.1)

st.markdown("</div>", unsafe_allow_html=True)

# ---- Predict ----
if st.button("🏡  Predict House Price"):
    raw = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    scaled = scaler.transform(raw)
    price = model.predict(scaled)[0]
    price_usd = price * 100000

    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:0.85rem; color:rgba(255,200,100,0.6); text-transform:uppercase; letter-spacing:1px;">Estimated Price</div>
        <div class="big-price">${price_usd:,.0f}</div>
        <div style="font-size:0.9rem; color:rgba(255,200,150,0.7); margin-top:8px;">
            Based on <strong>{model_choice}</strong> · R² = {r2:.4f} · RMSE = ${rmse*100000:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Actual vs Predicted ----
st.markdown('<div class="glass" style="margin-top:20px;">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📈 Actual vs Predicted (Sample)</div>', unsafe_allow_html=True)
sample_idx = np.random.choice(len(y_test), 80, replace=False)
y_t_s = np.array(y_test)[sample_idx]
y_p_s = y_pred[sample_idx]
chart_df = pd.DataFrame({"Actual": y_t_s * 100000, "Predicted": y_p_s * 100000})
st.line_chart(chart_df, color=["#ff9500", "#60a5fa"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>🔢 California Housing Dataset · Scikit-Learn · Streamlit</div>", unsafe_allow_html=True)
