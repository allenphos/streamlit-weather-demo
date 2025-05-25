import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load trained model and preprocessors ===
bundle = joblib.load("models/aussie_rain.joblib")

st.title("☔️ Прогноз дощу на завтра (Австралія)")

model      = bundle["model"]
imputer    = bundle["imputer"]
scaler     = bundle["scaler"]
encoder    = bundle["encoder"]
input_cols = bundle["input_cols"]
num_cols   = bundle["numeric_cols"]
cat_cols   = bundle["categorical_cols"]
enc_cols   = bundle["encoded_cols"]

st.image('images/aus_cloud_vis_20240923.gif')

st.header("Введіть погодні параметри:")

# Дефолтні значення та підказки (заповніть згідно ваших даних)
test_input  = {'Date': '2021-06-19',
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
    "MinTemp": "Мінімальна температура за добу (°C)",
    "MaxTemp": "Максимальна температура за добу (°C)",
    "Rainfall": "Опади за добу (мм)",
    "Evaporation": "Випаровування води за добу (мм)",
    "Sunshine": "Години сонячного сяйва за добу",
    "WindGustSpeed": "Швидкість пориву вітру (км/год)",
    "WindSpeed9am": "Швидкість вітру о 9:00 (км/год)",
    "WindSpeed3pm": "Швидкість вітру о 15:00 (км/год)",
    "Humidity9am": "Вологість повітря о 9:00 (%)",
    "Humidity3pm": "Вологість повітря о 15:00 (%)",
    "Pressure9am": "Атмосферний тиск о 9:00 (гПа)",
    "Pressure3pm": "Атмосферний тиск о 15:00 (гПа)",
    "Cloud9am": "Хмарність о 9:00 (0-8)",
    "Cloud3pm": "Хмарність о 15:00 (0-8)",
    "Temp9am": "Температура о 9:00 (°C)",
    "Temp3pm": "Температура о 15:00 (°C)",
    "Location": "Оберіть метеостанцію",
    "WindGustDir": "Напрям пориву вітру",
    "WindDir9am": "Напрям вітру о 9:00",
    "WindDir3pm": "Напрям вітру о 15:00",
    "RainToday": "Чи був дощ сьогодні?",
}

user_input = {}

slider_cols = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm',
               'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Cloud9am', 'Cloud3pm']

for col in input_cols:
    if col in num_cols:
        allow_nan = st.checkbox(f"Немає даних по {col}?", key=f"nan_{col}")
        if allow_nan:
            user_input[col] = np.nan
        else:
            # Діапазони — підлаштуйте під свою задачу
            if col.startswith("Cloud"):
                user_input[col] = st.slider(
                    f"{col}", 0, 8, int(test_input.get(col, 4)), help=help_texts.get(col, "")
                )
            elif "Humidity" in col:
                user_input[col] = st.slider(
                    f"{col} (%)", 0, 100, int(test_input.get(col, 60)), help=help_texts.get(col, "")
                )
            elif "Temp" in col:
                user_input[col] = st.slider(
                    f"{col} (°C)", -10.0, 50.0, float(test_input.get(col, 10.0)), step=0.1, help=help_texts.get(col, "")
                )
            elif "Wind" in col:
                user_input[col] = st.slider(
                    f"{col} (км/год)", 0.0, 150.0, float(test_input.get(col, 10.0)), step=1.0, help=help_texts.get(col, "")
                )
            elif "Rain" in col or "Evaporation" in col:
                user_input[col] = st.slider(
                    f"{col} (мм)", 0.0, 50.0, float(test_input.get(col, 0.0)), step=0.1, help=help_texts.get(col, "")
                )
            else:
                user_input[col] = st.slider(
                    f"{col}", 0.0, 50.0, float(test_input.get(col, 0.0)), step=0.1, help=help_texts.get(col, "")
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

if st.button("Прогнозувати дощ завтра"):
    try:
        # Імпутація
        X_num = imputer.transform(user_data[num_cols])
        # Масштабування
        X_num = scaler.transform(X_num)
        # Кодування категорій
        X_cat = encoder.transform(user_data[cat_cols])
        if hasattr(X_cat, 'toarray'):
            X_cat = X_cat.toarray()
        # Об'єднання
        X_proc = np.hstack([X_num, X_cat])
        # Прогноз
        pred = model.predict(X_proc)[0]
        prob = model.predict_proba(X_proc)[0][1]
        st.markdown(
            f"**Прогноз:** {'☔ До́щ' if pred == 'Yes' else '⛅ Без дощу'}\n\n"
            f"**Ймовірність дощу:** {prob*100:.1f}%"
        )
        st.caption("Пояснення: модель враховує всі введені параметри. Чим ближче ймовірність до 100%, тим вища впевненість у прогнозі дощу.")
    except Exception as e:
        st.error(f"Виникла помилка: {e}")
        st.info("Перевірте, чи всі параметри введені коректно. Якщо помилка повторюється — зверніться до розробника.")
