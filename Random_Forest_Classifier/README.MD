# 🍷 Wine Quality Prediction Using Random Forest Classification

## 📌 Project Overview

This project predicts the **quality of red wine** based on its **chemical properties** using a **Random Forest Classification** model.  
The goal is to classify wine into **Good Quality** or **Low Quality** using machine learning.

The project demonstrates:
- Data preprocessing
- Feature selection
- Random Forest model training
- Model evaluation
- Feature importance analysis
- Interactive prediction system

---

## 🎯 Problem Statement

Wine quality is traditionally evaluated by **human tasters**, which can be subjective and time-consuming.  
This project aims to **automate wine quality prediction** using measurable chemical properties.

### Objective:
- Classify wine as **Good Quality** or **Low Quality**
- Identify which chemical features influence quality the most

---

## 📂 Dataset Description

- **Dataset Name:** Wine Quality – Red Wine
- **Source:** UCI Machine Learning Repository
- **Data Type:** Structured tabular data

### Target Variable
```text
quality ≥ 7  → Good Quality (1)
quality < 7  → Low Quality (0)
```
### Features Used
```
 Alcohol
 Sulphates
 Volatile Acidity
 pH
```

### Project Workflow

```
1. Load the dataset
2. Perform exploratory data analysis
3. Convert wine quality into binary labels
4. Select important chemical features
5. Split data into training and testing sets
6. Train the Random Forest model
7. Make predictions on test data
8. Evaluate model performance
9. Visualize confusion matrix
10. Analyze feature importance

```
###  Technologies Used

```
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Streamlit

```



