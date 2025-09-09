# flood-risk-ml-model
🌊 Flood Risk Prediction System using ML 🌊 A Random Forest–based model that predicts flood risk (Low, Medium, High) from weather and geographic factors. Supports manual input, CSV files, and live API data. Built to explore how AI can assist in disaster risk management.
# 🌊 Flood Risk Prediction System 🌊

A machine learning system to predict flood risks using synthetic data, weather APIs, and Random Forest classification.

## 🚀 Features
- Train flood risk prediction model with synthetic or real data
- Predict flood risk from:
  - Manual input
  - CSV file
  - Live weather API
- Interactive CLI for city-based predictions
- Saves trained models for reuse

## 📂 Project Structure
- `src/` → Core classes (Trainer, Predictor, Weather API, Utils)
- `data/` → Datasets (e.g., training data)
- `model/` → Saved trained models
- `app.py` → Main entry point

## ⚙️ Installation
```bash
git clone https://github.com/your-username/flood-risk-model.git
cd flood-risk-model
pip install -r requirements.txt
