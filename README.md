# 🌾 Crop Recommendation System

A machine learning web app that recommends the most suitable crop to grow based on soil nutrients and weather conditions , historical data.

🔗 **Live App:** https://croprecommender-t5zntxdcyevvbwveakvpkb.streamlit.app/

## Problem Statement

Choosing the right crop for the given soil and climate is a key decision for farmers. This project uses a machine learning model to suggest a crop from 7 input parameters, as a data-driven aid for that decision.

## Dataset

- Source: [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) 
- 2,200 samples, 22 crop classes (rice, maize, chickpea, mango, coffee, etc.)
- Features: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, rainfall

## Approach

1. **EDA:** checked distributions, class balance and feature relationships (pairplot)
2. **Preprocessing:** 80/20 train-test split, `StandardScaler` for features (fit on train only), `LabelEncoder` for target
3. **Model selection:** used **Optuna** (100 trials, 3-fold CV) to tune and compare SVM, Random Forest, Gradient Boosting and Logistic Regression
4. **Final model:** Random Forest Classifier
5. **Deployment:** model, scaler and encoder saved with `joblib` and served through a **Streamlit** app on Streamlit Community Cloud

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.30%** |


## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Optuna, Streamlit, joblib

## Project Structure

```
├── app.py                # Streamlit app
├── data.csv              # Dataset
├── rf_model.pkl          # Trained Random Forest model
├── scaler.pkl            # Fitted StandardScaler
├── label_encoder.pkl     # Fitted LabelEncoder
├── requirements.txt      # Dependencies
└── CRP_PRO.ipynb        # Training notebook 
```

## Run Locally

```bash
git clone https://github.com/abhi01-sys/<Crop_Recommender_>.git
cd <Crop_Recommender_>
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- The dataset is a clean, curated Kaggle dataset, so the 99% accuracy may not carry over to messy real-world field data.
- The model only recommends from the 22 crops present in the training data.

## Future Work

- Add yield prediction
- Add crop disease detection from leaf images using a CNN

## Author

**Abhishek Nagar**, B.Tech Artificial Intelligence, MITS Gwalior
GitHub: [abhi01-sys](https://github.com/abhi01-sys)
