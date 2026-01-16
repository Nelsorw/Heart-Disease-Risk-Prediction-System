# CHUB Heart Disease Risk Prediction System

## 1. Project Overview
An intelligent clinical decision support system to predict heart disease risk levels for patients at CHUB hospital. It uses routinely collected clinical, demographic, and diagnostic data to predict 5 classes of heart disease risk.

## 2. Features
- Predict heart disease risk based on 13 patient features.
- Returns predicted class and per-class probability distribution.
- Interactive frontend with color-coded results.
- Fully responsive design (desktop/tablet/mobile).
- Preprocessing + trained model integrated with Flask API.

## 3. Dataset
- Heart disease dataset with 5000 patient records.
- 13 features: age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak, slope, number of major vessels, thalassemia.
- Target: 5 classes (No Disease, Very Mild, Mild, Severe, Immediate Danger).

## 4. Installation
```bash
# Create virtual environment
python -m venv ITLML_801_S_A_25RP20037

# Activate venv (Windows)
ITLML_801_S_A_25RP20037\Scripts\activate


# Install dependencies
pip install -r requirements.txt
