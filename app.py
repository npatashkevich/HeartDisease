import streamlit as st
import numpy as np
import joblib
import pandas as pd
import sklearn
sklearn.set_config(transform_output='pandas')
from sklearn.preprocessing import OneHotEncoder

# Загрузка модели
ml_pipeline = joblib.load('ml_pipeline.pkl')


# Функция для получения предсказания
def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])
    return ml_pipeline.predict(input_df)


# Интерфейс Streamlit
st.title("Heart Disease Prediction")

# Поля ввода для пользователя
age = st.number_input("Age", min_value=0, max_value=150, step=1)
sex = st.radio("Sex", options=["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, step=1)
cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=1000, step=1)
fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
resting_ecg = st.selectbox("Resting Electrocardiogram Results", options=["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, step=1)
exercise_angina = st.radio("Exercise-Induced Angina", options=["No", "Yes"])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0, max_value=10, step=1)
st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])


# # Преобразование к числовым значениям
sex = 'M' if sex == "Male" else 'F'
chest_pain_mapping = {"Typical Angina": 'TA', "Atypical Angina": 'ATA', "Non-Anginal Pain": 'NAP', "Asymptomatic": 'ASY'}
fasting_bs = 1 if fasting_bs == "Yes" else 0
exercise_angina = 'Y' if exercise_angina == "Yes" else 'N'

# Кнопка для предсказания
if st.button("Predict"):
    input_data = np.array([age, sex, chest_pain_mapping[chest_pain_type], resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope])
    result = predict(input_data)
    if result == 1:
        st.write("Prediction: Heart Disease")
    else:
        st.write("Prediction: Normal")
