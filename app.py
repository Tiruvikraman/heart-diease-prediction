import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('heart_disease_model.h5')

# Load the scaler
# You need to fit the scaler with your training data here
# For example: scaler.fit(X_train)

st.title('Heart Disease Prediction')

st.write("""
This app predicts the probability of heart disease based on several input features.
Please enter the following information:
""")

# Create input fields for each feature
age = st.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Create a dictionary to map categorical variables to numerical values
categorical_map = {
    'Sex': {'Male': 1, 'Female': 0},
    'Chest Pain Type': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3},
    'Fasting Blood Sugar > 120 mg/dl': {'No': 0, 'Yes': 1},
    'Resting ECG Results': {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2},
    'Exercise Induced Angina': {'No': 0, 'Yes': 1},
    'Slope of the Peak Exercise ST Segment': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2},
    'Thalassemia': {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
}

if st.button('Predict'):
    # Convert categorical variables to numerical
    sex = categorical_map['Sex'][sex]
    cp = categorical_map['Chest Pain Type'][cp]
    fbs = categorical_map['Fasting Blood Sugar > 120 mg/dl'][fbs]
    restecg = categorical_map['Resting ECG Results'][restecg]
    exang = categorical_map['Exercise Induced Angina'][exang]
    slope = categorical_map['Slope of the Peak Exercise ST Segment'][slope]
    thal = categorical_map['Thalassemia'][thal]

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Scale the input data

    # Make prediction
    prediction = model.predict(input_data)
    probability = prediction[0][0]

    st.write(f'The probability of heart disease is: {probability:.2%}')
    
    if probability > 0.5:
        st.write('Based on the model prediction, there is a higher risk of heart disease. Please consult with a healthcare professional for a thorough evaluation.')
    else:
        st.write('Based on the model prediction, the risk of heart disease is lower. However, always maintain a healthy lifestyle and consult with healthcare professionals for regular check-ups.')

st.write("""
Note: This prediction is based on a machine learning model and should not be considered as a medical diagnosis. 
Always consult with a qualified healthcare professional for medical advice and proper diagnosis.
""")