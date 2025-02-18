import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.header("Diabetes Prediction App")

# Getting the input data from USER
Pregnancies = st.number_input("Number of Pregnancies")
Glucose = st.number_input("Level of Glucose")
BloodPressure = st.number_input("Blood Pressure")
SkinThickness = st.number_input("Skin Thickness")
Insulin = st.number_input("Level of Insulin")
BMI = st.number_input("Body Mass Index")
DiabetesPedigreeFunction = st.number_input("Diabetes Ratio")
Age = st.number_input("Age")

# Code for prediction
diab_diagnosis = ''

# Prediction
if st.button("Get Results"):
    # Create a NumPy array from the inputs
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Apply the scaler transformation
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    diab_pred = diabetes_model.predict(input_data_scaled)

    # Display the result    
    if diab_pred[0] == 0:
        diab_diagnosis = "The person is NOT diabetic."
        st.success(diab_diagnosis)
    else:
        diab_diagnosis = "The person is diabetic."
        st.error(diab_diagnosis)
