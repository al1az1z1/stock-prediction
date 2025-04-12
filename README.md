
# Stock Price Movement Prediction Using Logistic Regression

This project is part of the final team project for the course **Introduction to Artificial Intelligence (AAI 501)** at the University of San Diego. The objective is to apply AI and machine learning techniques to predict the next-day price movement (up or down) of publicly traded companies in the Information Technology sector, using historical stock price data from the New York Stock Exchange (NYSE).

---

## Problem Statement

Stock price movement prediction is a classic challenge in financial analytics and AI. This project investigates whether historical stock data can be used to predict the **next-day direction** of stock prices (up or down) for companies in the Information Technology sector. We aim to evaluate how effectively logistic regression can identify short-term trends, and whether company-specific or cross-company patterns yield better predictive power.

---

## Dataset

- **Source**: Kaggle – NYSE Stock Prices
- **Files Used**:
  - `prices-split-adjusted.csv` – historical stock prices
  - `securities.csv` – metadata including GICS sector info

We filtered 10 Information Technology companies (including AAPL) for the modeling task.

---

## Target Variable

The models predict whether the **closing price of a company will increase the next trading day**:
- `1` → Price goes up
- `0` → Price stays the same or goes down

---

## Experiment Design

- **Model 1**: Logistic Regression trained on one company’s data (AAPL)  
- **Model 2**: Logistic Regression trained on a combined dataset of 10 IT sector companies  
- **Comparison Objective**: Evaluate if including similar companies improves predictive accuracy  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC  
- **Target Variable**: Binary label indicating if the closing price increases the next day

---

## Key Steps

- Preprocessing and feature engineering using rolling averages, daily returns, and volume changes
- Time-based train-test split (80% train, 20% test)
- Independent evaluation of both models using metrics like Accuracy, Precision, Recall, F1, and ROC AUC
- Visual comparison using ROC curves and confusion matrices
- Optional extended experiment: repeat comparison across 5 companies to test generalizability

---

## Getting Started

To run the notebook and replicate the results:

```bash
git clone https://github.com/your-repo-url
cd stock-prediction
pip install -r requirements.txt
```

Then open `stock-prediction.ipynb` or `model_pipeline.ipynb` to begin.

---

## Team Members

- **Jack Kim**
  - GitHub: [JackKim123](https://github.com/jackkim-usd)
  - LinkedIn: [Jack Kim](https://www.linkedin.com/in/)
- **Mustafa Yunus**
  - GitHub: [mustafayunus](https://github.com/Mustafayunus3099-ui)
  - LinkedIn: [Mustafa Yunus](https://www.linkedin.com/in/)
- **Ali Azizi** 
  - GitHub: [al1az1z1](https://github.com/al1az1z1)
  - LinkedIn: [Ali Azizi](https://www.linkedin.com/in/al1az1z1)

---

## License & Academic Use

This project was developed as part of the course **AAI 501 – Introduction to Artificial Intelligence** at the **University of San Diego**.  
It is intended for educational purposes only and is released under the **MIT License**.  
All datasets are publicly available and sourced from Kaggle. Code contributions follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/).
