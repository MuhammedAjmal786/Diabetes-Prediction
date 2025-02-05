import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Diabetes Prediction')

Pregnancies=st.number_input('Pregnancies')
Glucose=st.number_input('Glucose')
BloodPressure=st.number_input('BloodPressure')
SkinThickness=st.number_input('SkinThickness')
Insulin=st.number_input('Insulin')
BMI=st.number_input('BMI')
DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction')
Age=st.number_input('Age')

input_data={
        'Pregnancies':Pregnancies,
        'Glucose':Pregnancies,
        'BloodPressure':Pregnancies,
        'SkinThickness':SkinThickness,
        'Insulin':Insulin,
        'BMI':BMI,
        'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
        'Age':Age
    }
input_data=pd.DataFrame([input_data])
input_data['p_Age']=np.log1p(input_data['Age'])
input_data=input_data.drop('Age',axis=1)

model=pickle.load(open('model.pkl','rb'))
prediction=model.predict(input_data)[0]
if st.button('Submit'):
    if prediction==0:
            st.success("The patient does not have diabetes.")
    else:
        st.error("The patient has diabetes.")