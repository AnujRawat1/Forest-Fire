import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regressor model & Standard Scaler Model 
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Main Page to Predict Our data 
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Feature Scaling 
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', result=result[0])

        except Exception as e:
            # Log the exception or handle it gracefully
            print(f"Error occurred: {str(e)}")
            return render_template('home.html', result='Error occurred')

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
