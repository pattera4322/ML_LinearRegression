# Import necessary libraries
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the pre-trained model and feature names
model = joblib.load('linear_regression_model.pkl')  # Load your model file

# Define a function to make predictions
def predict_charges(age, sex_num, bmi, smoker):
    input_data = [[age, sex_num, bmi, smoker]]
    predicted_charges = model.predict(input_data)
    return predicted_charges[0]

# Load the coefficients of the model
coefficients = model.coef_
intercept = model.intercept_

# Create the Streamlit web app
st.title("Insurance Charges Prediction :syringe:")
st.sidebar.header("Input Features")

# Input fields for the user to enter data
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.radio("Sex", ["Female", "Male"])
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
smoker = st.sidebar.radio("Smoker", ["No", "Yes"])

# Map categorical values to numerical values
sex_num = 0 if sex == "Female" else 1
smoker_num = 0 if smoker == "No" else 1

# When the user clicks the "Predict" button
if st.button("Predict"):
    input_data = [[age, sex_num, bmi, smoker_num]]
    predicted_charge = predict_charges(age, sex_num, bmi, smoker_num)
    st.write(f":chart_with_upwards_trend: Predicted Charges: **${predicted_charge:.2f}**")

    # Calculate and display metrics
    st.markdown("**Linear Regression Model Metrics:**")
    st.write(f"Coefficients: {coefficients}")
    st.write(f"Intercept: {intercept}")

    # Optionally, calculate and display Mean Squared Error and R2 Score
    # Load actual data (replace with your data)
    actual_data = {
        "Age": [30, 35, 40, 45, 50],
        "Charges": [2500, 3200, 3900, 4200, 5200]
    }

    actual_charges = actual_data["Charges"]
    predicted_charges = [predict_charges(age, sex_num, bmi, smoker_num) for age in actual_data["Age"]]
    
    mse = mean_squared_error(actual_charges, predicted_charges)
    r2 = r2_score(actual_charges, predicted_charges)

    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R2) Score: {r2:.2f}")

    # Create a scatter plot of actual data and the regression line (similar to previous examples)
    plt.scatter(actual_data["Age"], actual_data["Charges"], label="Actual Data")
    ages = np.array([min(actual_data["Age"]), max(actual_data["Age"])])
    charges = coefficients[0] * ages + intercept
    plt.plot(ages, charges, color='red', label="Linear Regression")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.legend()
    st.pyplot(plt)

# To run the app, use the following command in your terminal
# streamlit run your_app.py
