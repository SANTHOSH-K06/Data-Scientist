# 🧠 Data Science & Machine Learning Portfolio

A curated collection of end-to-end data science projects covering supervised learning, unsupervised learning, and deployment-ready ML applications.

---

## 📁 Project Structure

```
Data-Scientist/
├── Random_Forest_Classifier/     # 🍷 Wine Quality Prediction App
│   ├── app.py                    # Streamlit app (Enhanced UI)
│   ├── random_forest_wine.pkl    # Trained model
│   ├── requirements.txt
│   └── wine_quality_prediction.ipynb
│
├── svm_digits_app/               # ✍️ Handwritten Digit Recognition App
│   ├── app.py                    # Streamlit app (Enhanced UI)
│   ├── requirements.txt
│   └── digits_svm_project.ipynb
│
└── Demo.py.txt                   # General ML demo script
```

---

## 🚀 Applications

### 🍷 Wine Quality Predictor (`Random_Forest_Classifier/`)
- **Algorithm:** Random Forest Classifier
- **Task:** Binary classification — High Quality vs Low Quality wine
- **Input features:** Alcohol, Sulphates, Volatile Acidity, pH
- **Frontend:** Premium dark theme with glassmorphism, animated results, confidence bars
- **Deploy:**
  ```bash
  cd Random_Forest_Classifier
  pip install -r requirements.txt
  streamlit run app.py
  ```

### ✍️ Handwritten Digit Recognizer (`svm_digits_app/`)
- **Algorithm:** Support Vector Machine (Linear / RBF kernel)
- **Task:** Multi-class classification — Digits 0–9
- **Dataset:** scikit-learn built-in Digits dataset (8×8 pixel images)
- **Frontend:** Premium dark blue theme, probability chart, real-time prediction
- **Deploy:**
  ```bash
  cd svm_digits_app
  pip install -r requirements.txt
  streamlit run app.py
  ```

---

## 🤖 Supervised Learning Topics Covered

| Category | Methods |
|---|---|
| **Regression** | Linear Regression, Lasso, Ridge, House Price Prediction |
| **Classification** | Logistic Regression, SVM, Random Forest, Naive Bayes, Decision Tree |
| **Ensemble** | Random Forest, Gradient Boosting |

## 📊 Unsupervised Learning

- K-Means Clustering
- DBSCAN (Density-Based Clustering)

## 🛠 ML Engineering

- Bias–Variance Tradeoff
- K-Fold Cross Validation
- GridSearchCV / RandomizedSearchCV
- Feature Selection (Filter, Wrapper, Embedded)
- Model Serialization (Joblib)
- Streamlit Deployment

---

## 🧪 Tech Stack

`Python` &nbsp; `NumPy` &nbsp; `Pandas` &nbsp; `Matplotlib` &nbsp; `Seaborn` &nbsp; `Scikit-Learn` &nbsp; `Streamlit` &nbsp; `Joblib`

---

## 📌 Upcoming Additions

- [ ] XGBoost Advanced Tuning
- [ ] Deep Learning Projects
- [ ] NLP Applications
- [ ] Model Monitoring
- [ ] Docker Deployment
- [ ] CI/CD for ML Apps

---

*Built with ❤️ as part of a Data Science specialization journey.*
