# End-to-End E-commerce Analytics Platform 🛍️

![Project Banner](https://img.shields.io/badge/Status-Complete-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![ML](https://img.shields.io/badge/ML-Production_Ready-orange)

## 🎯 Project Overview

A comprehensive machine learning platform analyzing 112,650+ Brazilian e-commerce transactions to drive business insights through predictive modeling, customer analytics, and experimental design. Built 4 production-ready ML models achieving exceptional performance across pricing, logistics, and customer retention.

## 🚀 Key Results

| Model | Metric | Performance | Business Impact |
|-------|---------|-------------|-----------------|
| **Price Prediction** | R² Score | 62% | Revenue optimization |
| **Delivery Optimization** | R² Score | 36% | Logistics efficiency |
| **Order Classification** | Accuracy | 98% | Operations automation |
| **Churn Prediction** | Accuracy | 100% | Customer retention |

### 🔬 A/B Testing Insights
- **Payment Method Analysis**: Credit card users spend 23% more than boleto users
- **Statistical Significance**: p-value < 0.05 across all experiments

## 🏗️ System Architecture

```bash
Raw Data (112K+ transactions)
↓
Data Preprocessing & Feature Engineering
↓
┌─────────────────────────────────────────┐
│           ML Pipeline                   │
├─────────────┬─────────────┬─────────────┤
│ Regression  │ Classification│ A/B Testing │
│ Models      │ Models        │ Framework   │
└─────────────┴─────────────┴─────────────┘
↓
```

## Business Intelligence Dashboard

## 📊 Dataset & Features

**Source**: Olist Brazilian E-commerce Dataset (Kaggle)
- **Orders**: 112,650 transactions
- **Customers**: 98,666 unique customers  
- **Time Period**: 2016-2018
- **Features**: 30+ engineered features including RFM analysis

### 🔧 Feature Engineering
- **Temporal Features**: Order timing, seasonality, weekend patterns
- **Geographic Features**: State-level delivery analysis
- **Customer Behavior**: RFM (Recency, Frequency, Monetary) analysis
- **Product Features**: Category, weight, dimensions, pricing

## 🤖 Machine Learning Models

### 1. Price Prediction (Regression)
```python
Models Compared: Linear Regression, Random Forest, XGBoost
Best Performer: Random Forest (R² = 0.62)
Key Features: freight_value, product_description_length, product_weight
```

### 2. Delivery Time Prediction (Regression)
```python
Models Compared: Linear Regression, Random Forest, XGBoost  
Best Performer: Random Forest (R² = 0.36)
Key Features: customer_zip_code, freight_value, product_weight
```
### 3.  Order Status Classification
```python
Models Compared: Logistic Regression, Random Forest, XGBoost
Best Performer: Random Forest (98% Accuracy)
Target: Binary classification (Delivered vs Not-Delivered)
```

### 4. Customer Churn Prediction
```python
Models Compared: Logistic Regression, Random Forest, XGBoost
Best Performer: Random Forest (100% Accuracy)
Churn Definition: RFM-based business logic (40.6% churn rate)
```

### 📈 Business Analytics
Customer Segmentation

High-Value Customers: Recent purchases, high frequency, premium spending
At-Risk Customers: 355+ days recency, single purchases, <$46 spending

Geographic Insights

Top Revenue States: SP, RJ, MG
Delivery Challenges: Remote areas show 2x longer delivery times

Payment Behavior

Credit Card Users: Higher order values, better retention
Boleto Users: Price-sensitive, seasonal patterns

### A/B testing Framework 
Payment Method Experiment
```python
# Group A: Credit Card Users (n=45,123)
# Group B: Boleto Users (n=19,784)
# Metric: Average Order Value
# Result: Statistically significant difference (p<0.001)
```
### Key Findings
- Credit card users have 23% higher average order value
- Weekend orders show different payment preferences
- Regional variations in payment method adoption

### 🛠️ Technology Stack

- Languages: Python 3.8+
- ML Libraries: Scikit-learn, XGBoost, Pandas, NumPy
- Visualization: Matplotlib, Seaborn, Plotly
- Statistical Analysis: SciPy, Statsmodels
- Data Processing: Pandas, NumPy
- Development: Jupyter Notebooks, Git

### 📁 Repository Structure
```python
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Source code modules
├── data/               # Data files (gitignored)
├── visualizations/     # Generated plots and charts
├── models/             # Saved model files
└── requirements.txt    # Dependencies
```
### 🚀 Getting Started
```python
Prerequisites

Python 3.8+
Jupyter Notebook
Git
```

### Installation
```python
#clone repository 
git clone : https://github.com/Modupeolawuraola/ML-E-commerce-End-to-end-Analytics

```

### Install dependencies 
```python
pip install -r requirements.txt

#download data kaggle API required 
kaggle datasets download - d olistbr/brazillian-ecommerce

```

### QuickStart 
```python

# Load and explore data
python src/data_preprocessing.py

# Run price prediction model
jupyter notebook notebooks/03_price_prediction.ipynb

# Execute full pipeline
python src/main.py
```

### 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


### 👤 Author
Modupeola Fagbenro

- GitHub: @Modupeolawuraola
- 
- LinkedIn: https://www.linkedin.com/in/modupeola-fagbenro/








