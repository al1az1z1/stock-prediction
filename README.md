# Stock Price Movement Prediction Using Logistic Regression

This project is part of the final team project for the course **Introduction to Artificial Intelligence (AAI 501)** at the University of San Diego. The objective is to apply AI and machine learning techniques to predict the next-day price movement (up or down) of publicly traded companies in the Information Technology sector, using historical stock price data from the New York Stock Exchange (NYSE).

We explore and compare two logistic regression models:

- **Model 1**: Trained on data from a single company (e.g., AAPL) to capture its individual behavior.
- **Model 2**: Trained on the same company plus 9 additional companies from the same sector to explore whether incorporating similar businesses improves prediction accuracy.

The goal is to determine whether company-specific patterns or cross-company trends lead to more accurate predictions.

---

##  Dataset

- **Source**: Kaggle – NYSE Stock Prices
- **Files Used**:
  - `prices-split-adjusted.csv` – historical stock prices
  - `securities.csv` – metadata including GICS sector info

We filtered 10 Information Technology companies (including AAPL) for the modeling task.

---

##  Target Variable

The models predict whether the **closing price of a company will increase the next trading day**:
- `1` → Price goes up
- `0` → Price stays the same or goes down

---

##  Key Steps

- Preprocessing and feature engineering using rolling averages, daily returns, and volume changes
- Time-based train-test split (80% train, 20% test)
- Independent evaluation of both models using metrics like Accuracy, Precision, Recall, F1, and ROC AUC
- Visual comparison using ROC curves and confusion matrices
- Optional extended experiment: repeat comparison across 5 companies to test generalizability

---

##  Team Members

- Member 1: [Your Name] – EDA, preprocessing
- Member 2: [Name] – Logistic Model 1 (single company)
- Member 3: [Name] – Logistic Model 2 (multi-company)

---

## 📄 License

This project was developed for academic purposes as part of the AAI 501 – Introduction to Artificial Intelligence course at the University of San Diego.  
It is open-sourced under the **MIT License**.
