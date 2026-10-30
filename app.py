import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("rf_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌾Crop Recommendation System")

st.subheader("📜 System Recommend Based on Historical Data")

@st.cache_data
def get_data():
    df = pd.read_csv('data.csv')
    return df

@st.cache_data
def convert_for_download(df):
    return df.to_csv(index=False).encode("utf-8")

df = get_data()
csv = convert_for_download(df)


st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
    icon=":material/download:",
)

st.link_button("GIT-HUB", "https://github.com/abhi01-sys")

with st.form("my_form"):

    N = st.number_input(
        "Nitrogen(N) Quantity : ", min_value=0.0, max_value=150.0, value=50.0, placeholder="Type a number..."
    )

    P = st.number_input(
        "Phosphorus(P) Quantity : ", min_value=5.0, max_value=180.0, value=55.0, placeholder="Type a number..."
    )

    K = st.number_input(
        "Potassium(K) Quantity : ", min_value=5.0, max_value=250.0, value=50.0, placeholder="Type a number..."
    )

    temperature = st.number_input(
        "Temperature : ", min_value=10.0, max_value=45.0, value=25.0, placeholder="Type a number..."
    )

    humidity = st.number_input(
        "Humidity : ", min_value=10.0, max_value=100.0, value=70.0, placeholder="Type a number..."
    )

    ph = st.slider(
        "pH : ", 1.0, 10.0, 5.5)

    rainfall = st.number_input(
        "Rainfall : ", min_value=15.0, max_value=300.0, value=100.0, placeholder="Type a number..."
    )


    submitted = st.form_submit_button("Recommend")
    if submitted:
        data = np.array([N, P, K, temperature, humidity, ph, rainfall])

        data = data.reshape(1, -1)

        scaled_data = scaler.transform(data)

        predict = model.predict(scaled_data)

        val = encoder.inverse_transform(predict)

        st.success(f"Recommended Crop: {val[0]}")
