import streamlit as st
import pandas as pd
import numpy as np
#import joblib
import sys
st.write(sys.path)
try:
    import joblib
    st.write("Joblib imported OK")
except Exception as e:
    st.error(str(e))

import os

st.write("Current working directory:", os.getcwd())
st.write("Files in models folder:", os.listdir("models"))
st.write("Files in images folder:", os.listdir("images"))

# === Load trained model and preprocessors ===
import traceback
try:
    bundle = joblib.load("models/aussie_rain.joblib")
except Exception as e:
    import streamlit as st
    st.error(f"Exception during model loading: {e}")
    st.text(traceback.format_exc())
    bundle = None

st.title("☔️ Tomorrow's Rain Forecast (Australia)")

model      = bundle["model"]
imputer    = bundle["imputer"]
scaler     = bundle["scaler"]
encoder    = bundle["encoder"]
input_cols = bundle["input_cols"]
num_cols   = bundle["numeric_cols"]
cat_cols   = bundle["categorical_cols"]
enc_cols   = bundle["encoded_cols"]

st.image('images/aus_cloud_vis_20240923.gif')

st.header("Enter Weather Parameters:")

# Default values and tooltips (adjust based on your data)
test_input = {
    'Date': '2021-06-19',
    'Location': 'Katherine',
    'MinTemp': 23.2,
    'MaxTemp': 33.2,
    'Rainfall': 10.2,
    'Evaporation': 4.2,
    'Sunshine': 0.0,
    'WindGustDir': 'NNW',
    'WindGustSpeed': 52.0,
    'WindDir9am': 'NW',
    'WindDir3pm': 'NNE',
    'WindSpeed9am': 13.0,
    'WindSpeed3pm': 20.0,
    'Humidity9am': 89.0,
    'Humidity3pm': 58.0,
    'Pressure9am': 1004.8,
    'Pressure3pm': 1001.5,
    'Cloud9am': 8.0,
    'Cloud3pm': 5.0,
    'Temp9am': 25.7,
    'Temp3pm': 33.0,
    'RainToday': 'Yes'
}

help_texts = {
    "MinTemp": "Minimum temperature during the day (°C)",
    "MaxTemp": "Maximum temperature during the day (°C)",
    "Rainfall": "Total rainfall during the day (mm)",
    "Evaporation": "Total evaporation during the day (mm)",
    "Sunshine": "Hours of sunshine during the day",
    "WindGustSpeed": "Wind gust speed (km/h)",
    "WindSpeed9am": "Wind speed at 9:00 (km/h)",
    "WindSpeed3pm": "Wind speed at 15:00 (km/h)",
    "Humidity9am": "Humidity at 9:00 (%)",
    "Humidity3pm": "Humidity at 15:00 (%)",
    "Pressure9am": "Atmospheric pressure at 9:00 (hPa)",
    "Pressure3pm": "Atmospheric pressure at 15:00 (hPa)",
    "Cloud9am": "Cloud cover at 9:00 (0-8)",
    "Cloud3pm": "Cloud cover at 15:00 (0-8)",
    "Temp9am": "Temperature at 9:00 (°C)",
    "Temp3pm": "Temperature at 15:00 (°C)",
    "Location": "Select the weather station",
    "WindGustDir": "Wind gust direction",
    "WindDir9am": "Wind direction at 9:00",
    "WindDir3pm": "Wind direction at 15:00",
    "RainToday": "Did it rain today?",
}
min_value_dict = {
    "MinTemp": -10.0, "MaxTemp": -10.0, "Temp9am": -10.0, "Temp3pm": -10.0,
    "WindGustSpeed": 0.0, "WindSpeed9am": 0.0, "WindSpeed3pm": 0.0,
    "Rainfall": 0.0, "Evaporation": 0.0, "Sunshine": 0.0,
    "Humidity9am": 0.0, "Humidity3pm": 0.0,
    "Pressure9am": 980.0, "Pressure3pm": 980.0,
    "Cloud9am": 0, "Cloud3pm": 0,
}

max_value_dict = {
    "MinTemp": 50.0, "MaxTemp": 50.0, "Temp9am": 50.0, "Temp3pm": 50.0,
    "WindGustSpeed": 150.0, "WindSpeed9am": 100.0, "WindSpeed3pm": 100.0,
    "Rainfall": 100.0, "Evaporation": 50.0, "Sunshine": 15.0,
    "Humidity9am": 100.0, "Humidity3pm": 100.0,
    "Pressure9am": 1050.0, "Pressure3pm": 1050.0,
    "Cloud9am": 8, "Cloud3pm": 8,
}

user_input = {}

# Create input fields for each column
for col in input_cols:
    if col in num_cols:
        allow_nan = st.checkbox(f"No data for {col}?", key=f"nan_{col}")
        if allow_nan:
            user_input[col] = np.nan
        else:
            if col.startswith("Cloud"):
                user_input[col] = st.slider(
                    f"{col}", 0, 8, int(test_input.get(col, 4)), help=help_texts.get(col, "")
                )
            elif "Humidity" in col:
                user_input[col] = st.slider(
                    f"{col} (%)", 0, 100, int(test_input.get(col, 60)), help=help_texts.get(col, "")
                )
            elif col in min_value_dict:
                user_input[col] = st.slider(
                    f"{col}",
                    min_value_dict[col],
                    max_value_dict[col],
                    float(test_input.get(col, 0.0)),
                    step=0.1,
                    help=help_texts.get(col, "")
                )
            else:
                user_input[col] = st.slider(
                    f"{col}", 0.0, 100.0, float(test_input.get(col, 0.0)), step=0.1, help=help_texts.get(col, "")
                )
    elif col in cat_cols:
        if hasattr(encoder, 'categories_'):
            options = list(encoder.categories_[cat_cols.index(col)])
        else:
            options = ["Unknown"]
        default_idx = options.index(test_input[col]) if col in test_input and test_input[col] in options else 0
        user_input[col] = st.selectbox(
            f"{col}",
            options,
            index=default_idx,
            help=help_texts.get(col, "")
        )
    else:
        user_input[col] = st.text_input(
            f"{col}",
            value=test_input.get(col, ""),
            help=help_texts.get(col, "")
        )


user_data = pd.DataFrame([user_input])

if st.button("Predict rain tomorrow"):
    try:
        # Imputation
        X_num = imputer.transform(user_data[num_cols])
        # Scaling
        X_num = scaler.transform(X_num)
        # Encoding categorical features
        X_cat = encoder.transform(user_data[cat_cols])
        if hasattr(X_cat, 'toarray'):
            X_cat = X_cat.toarray()
        # Concatenate processed features
        X_proc = np.hstack([X_num, X_cat])
        # Prediction
        pred = model.predict(X_proc)[0]
        prob = model.predict_proba(X_proc)[0][1]
        st.markdown(
            f"**Prediction:** {'☔ Rain' if pred == 'Yes' else '⛅ No Rain'}\n\n"
            f"**Rain probability:** {prob*100:.1f}%"
        )
        st.caption("Note: The model takes into account all entered parameters. The closer the probability is to 100%, the higher the model's confidence in predicting rain.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check if all parameters are entered correctly. If the error persists, please contact the developer.")
