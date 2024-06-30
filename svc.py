import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import streamlit as st

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# Streamlit app


# Streamlit form for user inputs
st.title('Diabetes Prediction')

# Create a form for user input
with st.form("input_form"):
    st.write("Fill in the details to predict diabetes:")
    
    # Input fields for user to enter data
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("BloodPressure", min_value=0)
    skin_thickness = st.number_input("SkinThickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    diabetes_pedigree = st.number_input("DiabetesPedigreeFunction", min_value=0.000, step=0.0001)
    age = st.number_input("Age", min_value=0, step=1)

    # Submit button to make prediction
    submitted = st.form_submit_button("Predict")

    if submitted:
        X = df.drop(columns='Outcome')
        Y = df['Outcome']

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20, stratify=Y)

        # Standardize the training and testing data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the SVM model with hyperparameters
        svm_model = SVC(kernel='linear', C=1.0, random_state=42)
        svm_model.fit(X_train, Y_train)

        # Standardize the input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = svm_model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            st.error("It seems that the person is **diabetic**.")
        else:
            st.success("It seems that the person is **not diabetic**.")
            st.snow()
