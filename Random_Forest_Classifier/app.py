import streamlit as st
import numpy as np
import joblib
import os

# ---------------- Load Model ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_wine.pkl")
model = joblib.load(MODEL_PATH)

# ---------------- Page Config ----------------
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b0b0b, #1a0006);
        color: #f5f5f5;
    }
    h1, h2, h3 {
        color: #8b1e2d;
    }
    label {
        color: #e0c9a6 !important;
        font-weight: 600;
    }
    input {
        background-color: #1a1a1a !important;
        color: #f5f5f5 !important;
        border: 1px solid #8b1e2d !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #8b1e2d, #b22222);
        color: white;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }
    .prediction-box {
        background: #140006;
        padding: 25px;
        border-radius: 14px;
        border-left: 7px solid #b22222;
        margin-top: 25px;
    }
    .footer {
        color: #cfae70;
        font-size: 13px;
        text-align: center;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Title ----------------
st.title("🍷 Wine Quality Prediction")
st.write("Predict wine quality using a **Random Forest Classifier**")

st.markdown("---")

# ---------------- Input Fields ----------------
alcohol = st.number_input("Alcohol", min_value=0.0, value=10.0)
sulphates = st.number_input("Sulphates", min_value=0.0, value=0.65)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.5)
pH = st.number_input("pH", min_value=0.0, value=3.3)

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_data = np.array([[alcohol, sulphates, volatile_acidity, pH]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        quality = "High Quality"
        message = "This wine is pleasant in taste and can be consumed with ease."
        emoji = "🍷"
    else:
        quality = "Low Quality"
        message = "This wine may have poor taste and is not ideal for consumption."
        emoji = "⚠️"

    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>{emoji} Quality Rating: {quality}</h3>
            <p style="color:#e0c9a6; font-size:16px;">
                {message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>Random Forest Classification | Streamlit Deployment</div>",
    unsafe_allow_html=True
)