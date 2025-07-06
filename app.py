import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Energy Consumption Forecasting App - LSTM Model")

uploaded_file = st.file_uploader("Upload your cleaned time series CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    st.success("Data uploaded and parsed successfully.")
    st.line_chart(df['PJMW_MW'])

    steps = st.slider("Select forecast horizon (days):", min_value=7, max_value=90, value=30)

    if st.button("Run LSTM Forecast"):
        st.subheader("Forecast using LSTM Model")
        model = load_model("lstm_model.h5", compile=False)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['PJMW_MW']])
        lookback = 30
        input_seq = data_scaled[-lookback:].reshape(1, lookback, 1)

        predictions = []
        for _ in range(steps):
            pred = model.predict(input_seq, verbose=0)[0][0]
            predictions.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
        result = pd.DataFrame({"Date": future_dates, "Predicted": predicted_values})
        st.line_chart(result.set_index("Date"))
        st.success("Forecast complete using LSTM model.")
