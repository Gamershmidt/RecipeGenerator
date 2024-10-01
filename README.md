# Symptom2Disease: Multiclass Disease Prediction Model

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)

## Project Overview

**Symptom2Disease** is a machine learning project that aims to predict diseases based on symptoms provided by users. It is a multiclass classification problem where the input is a text about symptoms, and the output is the most probable disease from a predefined list of diseases.

This project incorporates the following key elements:
- **Model Development**: The model is trained using machine learning algorithms to predict the diseases based on symptom data.
- **Deployment**: The model is deployed using **Docker**, with **FastAPI** handling the backend API, and **Streamlit** providing the frontend interface.
- **Data Processing**: **DVC** (Data Version Control) is used for versioning and preprocessing of the data.
- **Pipeline Management**: The pipeline for model training and updates is managed using **Airflow**.

## Features
- **Multiclass classification**: Predict a disease from a list of possible diseases based on input symptoms.
- **FastAPI**: Provides a fast and lightweight REST API for interacting with the model.
- **Streamlit**: User-friendly frontend for users to input symptoms and view predicted diseases.
- **DVC**: For efficient data management, preprocessing, and version control.
- **Airflow**: Manages the data pipeline and automation of model retraining and updates.

## Tech Stack

### Model:
- **Machine Learning**: RNN model (torch)

### Backend:
- **FastAPI**: To serve the trained model via API.

### Frontend:
- **Streamlit**: Provides an easy-to-use frontend for users to input symptoms and receive disease predictions.

### Infrastructure:
- **Docker**: To containerize the application for easy deployment.

### Data Management and Preprocessing:
- **DVC**: Manages datasets and versioning.

### Pipeline:
- **Airflow**: Automates the end-to-end process from data ingestion, preprocessing, model training, and deployment.
