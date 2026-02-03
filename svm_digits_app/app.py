import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ---------------- Page config ----------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image of a digit (0–9)")

# ---------------- Load dataset ----------------
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- Sidebar: Model choice ----------------
st.sidebar.header("Model Selection")

model_type = st.sidebar.radio(
    "Choose Model Type",
    ["Linear SVM", "Non-Linear SVM (RBF)"]
)

# ---------------- Train model ----------------
if model_type == "Linear SVM":
    model = SVC(kernel="linear", C=10)
else:
    model = SVC(kernel="rbf", C=10, gamma="scale")

model.fit(X_train, y_train)

# ---------------- Model accuracy ----------------
accuracy = model.score(X_test, y_test)

# ---------------- Image upload ----------------
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    image = image.resize((8, 8))
    image_array = np.array(image)

    # Invert colors
    image_array = 255 - image_array

    # Normalize to match digits dataset
    image_array = (image_array / 255.0) * 16

    # Flatten
    image_array = image_array.reshape(1, -1)

    # Predict
    prediction = model.predict(image_array)[0]

    # Output
    st.subheader("Prediction Result")
    st.success(f"Predicted Digit: **{prediction}**")

    st.subheader("Model Performance")
    st.info(f"Model Used: **{model_type}**")
    st.info(f"Accuracy on Test Data: **{accuracy:.4f}**")