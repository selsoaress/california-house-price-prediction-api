# 🏠 California Housing Prices - MLOps Pipeline

An end-to-end Machine Learning workflow designed to predict housing prices in California. This project demonstrates a complete lifecycle: from data training and hyperparameter tuning with **Scikit-Learn**, to experiment tracking and model registry with **MLflow**, culminating in a real-time inference API built with **FastAPI**.

## 🚀 Key Features

- **Experiment Tracking:** Logs metrics (MAE, RMSE), parameters, and artifacts for every training run.
- **Model Registry:** Automatically versions models and manages stages.
- **REST API:** A robust API endpoint to consume the model using `Pydantic` for data validation.

## 🛠️ Tech Stack

- **Python 3.12+**
- **Machine Learning:** Scikit-Learn, Pandas
- **MLOps:** MLflow
- **API:** FastAPI, Uvicorn

## 📂 Project Structure

    ├── api.py           # FastAPI application
    ├── train.py         # Training script with MLflow tracking
    ├── mlflow.db        # SQLite database for MLflow registry
    ├── requirements.txt # Project dependencies
    ├── mlruns/          # Directory for MLflow artifacts
    └── sample_data/     # Dataset (California Housing)
