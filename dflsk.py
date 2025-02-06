from flask import  Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('dhml.html',Prediction_text='')

@app.route('/predict',methods=['POST'])
def predict():
    Pregnancies=float(request.form['Pregnancies'])
    Glucose=float(request.form['Glucose'])
    BloodPressure=float(request.form['BloodPressure'])
    SkinThickness=float(request.form['SkinThickness'])
    Insulin=float(request.form['Insulin'])
    BMI=float(request.form['BMI'])
    DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
    Age=float(request.form['Age'])

    input_data={
        'Pregnancies':Pregnancies,
        'Glucose':Glucose,
        'BloodPressure':BloodPressure,
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

    a=''
    if prediction==0:
        a="The patient does not have diabetes."
    else:
        a="The patient has diabetes."

    return render_template('dhml.html',Prediction_text=a)
if __name__=='__main__':
    app.run(debug=True)
