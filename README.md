# Delivery Time Prediction for Food Delivery Networks

<div align="center">
  <em>Predicting delivery times for restaurant orders using machine learning.</em>
</div>

<br>

<div align="center">
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/XGBoost-337ab7?style=for-the-badge&logo=xgboost&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
<img src="https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white">
</div>

## 📖 Project

### 👨🏻‍🏫 Introduction

This project develops a **predictive model** to estimate delivery times for orders in a food delivery restaurant network, using historical order data. The solution follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology and includes a complete **Streamlit web application** for interactive exploration and predictions.

The model aims to replace the current approach (historical average) with a more sophisticated machine learning solution that accounts for multiple variables affecting delivery time, such as store characteristics, order timing, and customer demand factors.

🌐 **Link**: The application is deployed on Google Cloud Run and accessible at:  
**[https://delivery-time-prediction-231487788414.southamerica-east1.run.app](https://delivery-time-prediction-231487788414.southamerica-east1.run.app)**

### 🎯 Goals

- **Business Understanding**: Define the delivery time prediction problem and business rules.
- **Data Understanding**: Perform exploratory data analysis to reveal insights about delivery patterns.
- **Data Preparation**: Clean and transform raw data, handling missing values and feature engineering.
- **Modeling**: Build and compare baseline and optimized XGBoost regression models.
- **Evaluation**: Assess model performance using MAE and RMSE metrics with cross-validation.
- **Deployment**: Deploy the solution as an interactive Streamlit application on Google Cloud Platform.

### 🔍 Problem Classification

**Supervised Machine Learning Regression** - Predicting continuous delivery time values based on historical order features.

### 📊 Dataset

- **Source**: iFood export with 90-day historical data (August 9, 2025 - November 7, 2025)
- **Initial Features**: 29 columns including store information, order details, timing, and geographical data
- **Target Variable**: Delivery time (minutes)
- **Data Dictionary**: Detailed field descriptions available in `data/raw/raw_data_dict.json`

### 🤖 Models

1. **Baseline Model**: Dummy Regressor with mean strategy
2. **Initial XGBoost**: XGBoost Regressor with 100 estimators
3. **Optimized XGBoost**: Hyperparameter tuning using Optuna with cross-validation

## 🗄 Notebooks

- [1.0-eda.ipynb](notebooks/1.0-eda.ipynb) - Exploratory Data Analysis
- [2.0-machine_learning.ipynb](notebooks/2.0-machine_learning.ipynb) - Model Development and Evaluation

## 🚀 Usage

### Local Development

Run the Streamlit application locally:

```bash
make run-local
```

### Docker

Build and run using Docker:

```bash
docker build -t delivery-time-prediction .
docker run -p 8501:8501 delivery-time-prediction
```

### Deployment to Google Cloud

Deploy to Google Cloud Run:

```bash
make deploy
```

This will:
- Build the Docker image using Cloud Build
- Push to Artifact Registry
- Deploy to Cloud Run in `southamerica-east1` region

## 🔬 Evaluation Metrics

- **Primary**: MAE (Mean Absolute Error)
- **Secondary**: RMSE (Root Mean Squared Error)
- **Validation**: 3-fold cross-validation on train/test sets
- **Analysis**: Learning curves and residual plots to detect overfitting/underfitting

## 🛠️ Technologies

- **Python 3.11** - Core programming language
- **Streamlit** - Interactive web application framework
- **scikit-learn** - Machine learning algorithms and preprocessing
- **XGBoost** - Gradient boosting for regression
- **Optuna** - Hyperparameter optimization
- **pandas** - Data manipulation and analysis
- **ydata-profiling** - Automated exploratory data analysis
- **yellowbrick**: Machine learning visualization and analysis
- **Docker** - Containerization
- **Google Cloud Platform** - Cloud deployment (Cloud Run, Cloud Build, Artifact Registry)

## 📦 Folder Structure

    ├── app.py                 <- Streamlit main application entry point
    ├── main.py                <- Python entry point
    ├── cloudbuild.yaml        <- Google Cloud Build configuration
    ├── Dockerfile             <- Docker container configuration
    ├── Makefile               <- Commands for running and deploying
    ├── pyproject.toml         <- Project dependencies and configuration
    │
    ├── data
    │   ├── raw                <- The original, immutable data dump
    │   │   └── raw_data_dict.json  <- Data dictionary with field descriptions
    │   └── processed          <- The final, canonical data sets for modeling
    │
    ├── models                 <- Trained models and model artifacts
    │
    ├── notebooks              <- Jupyter notebooks for analysis and development
    │   ├── 1.0-eda.ipynb
    │   └── 2.0-machine_learning.ipynb
    │
    ├── resources              <- Reports, figures, and visualizations
    │   ├── reports
    │   │   └── deliveries_data_profile_report.html
    │   └── visualizations
    │
    ├── src                    <- Source code for the project
    │   ├── __init__.py
    │   ├── configs            <- Configuration files
    │   │   └── settings.py
    │   └── streamlit          <- Streamlit application pages
    │       ├── business_understanding.py
    │       ├── data_understanding.py
    │       ├── data_preparation.py
    │       ├── modelling_and_evaluation.py
    │       ├── deployment.py
    │       └── playground.py  <- Interactive prediction interface
    │
    └── README.md              <- This file

## 📄 License

This project is part of academic work for the Pattern Recognition course (Computer Science Master's Degree).

---

<div align="center">
  Made with ❤️ for better delivery time predictions
</div>
