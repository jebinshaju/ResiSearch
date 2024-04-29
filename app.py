import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import joblib

# Load the trained model
model = joblib.load('uti_isolate_prediction_model.pkl')

# Load substitution patterns from isolates.reverse
with open("dataset/isolates.reverse", "r") as f:
    sed_commands = f.read().strip().split(';')  # Split commands by ';'

# Function to apply sed substitution to a string
def apply_sed_substitution(input_string, sed_command):
    sed_process = subprocess.Popen(['sed', sed_command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    sed_output, _ = sed_process.communicate(input=input_string)
    return sed_output.strip()

# Function to predict isolate based on input data
def predict_isolate(name, gender, age, location):
    # Map Gender and Location to indices
    gender_mapping = {'Male': 7, 'Female': 11}
    location_mapping = {'Urban': 85, 'Semi Urban': 68, 'Rural': 51}

    g = gender_mapping[gender]
    l = location_mapping[location]

    # Calculate input for prediction
    input_value = g * l
    x = np.array([[input_value]])

    # Predict
    if model:
        prediction = model.predict(x)[0]
        diff_probabilities = model.predict_proba(x)[0]

        # Apply sed substitution to prediction
        modified_prediction = str(prediction)
        for sed_command in sed_commands:
            modified_prediction = apply_sed_substitution(modified_prediction, sed_command)

        return input_value, modified_prediction, diff_probabilities
    else:
        return input_value, None, None

# Streamlit App
st.title('ResiSearch Solutions: AI-Powered Antibiotic Resistance Prediction')

# Company Information Sidebar
st.sidebar.title('About ResiSearch Solutions')
st.sidebar.image('resisearch_logo.png', caption='ResiSearch Solutions', use_column_width=True)
st.sidebar.write("ResiSearch Solutions is developing AI-powered software to determine antibiotic resistance before prescription. The company has already developed a Python-based tool for the same purpose")
# Founder Information
founder_name = "Freddy Alappattu"

st.sidebar.image("freddy.jpeg", caption=founder_name, use_column_width=True)

# Company Details
company_details = """
ResiSearch Solutions is a part of the incubation center of Nirmala College Muvattupuzha, dedicated to developing AI-powered software for antibiotic resistance prediction.
"""
st.sidebar.markdown(company_details)

# User Inputs
name = st.text_input('Name')
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', min_value=5, max_value=100, value=25, step=1)
location = st.selectbox('Location', ['Urban', 'Semi Urban', 'Rural'])

# Rerun prediction when inputs change
if st.button('Predict'):
    input_value, prediction, diff_probabilities = predict_isolate(name, gender, age, location)

    # Display results
    if prediction is not None:
        st.subheader('CDSS Report')
        st.write(f'**Patient:** {name}')
        st.write(f'**Gender:** {gender}')
        st.write(f'**Age:** {age}')
        st.write(f'**Location:** {location}')
        #st.write(f'**Genloc index:** {input_value}')
        st.write(f'**Most likely isolate expected:** {prediction}')

        # Display differential probabilities
        st.write('**Differential probabilities (refer pathogen encoding):**')
        for idx, prob in enumerate(diff_probabilities):
            st.write(f'Class {idx + 1}: {prob}')

        # Display disclaimer
        st.write('---')
        st.write('**Disclaimer:**')
        st.write('This is for academic purposes only.')
        st.write('The results must NOT be used for real patients without due testing, verification, and authorization by drug control authorities.')
        st.write('This tool is developed as part of ResiSearch Solutions\' efforts in antibiotic resistance research.')
