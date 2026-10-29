# 🌾 Crop Recommendation System

A machine learning web app that recommends the most suitable crop to grow based on soil nutrients and weather conditions.

🔗 **Live App:** https://croprecommender-t5zntxdcyevvbwveakvpkb.streamlit.app/

<!-- TODO: add a screenshot of your app here -->
<!-- ![App Screenshot](screenshot.png) -->

## Problem Statement

Choosing the right crop for the given soil and climate is a key decision for farmers. This project uses a machine learning model to suggest a crop from 7 input parameters, as a data-driven aid for that decision.

## Dataset

- Source: [Crop Recommendation Dataset (Kaggle)]( https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset )  
- 2,200 samples, 22 crop classes (rice, maize, chickpea, mango, coffee, etc.)
- Features: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, rainfall

## Approach

1. **EDA:** checked distributions, class balance and feature relationships (pairplot)
2. **Preprocessing:** 80/20 train-test split, `StandardScaler` for features (fit on train only), `LabelEncoder` for target
3. **Model : RandomForestClassifier()
