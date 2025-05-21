import numpy as np 
from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__, template_folder = 'template')

# Load models
with open('models/modelTOM.pkl', 'rb') as f:
    model = joblib.load('models/model2.pkl')

@app.route('/')
def home():
    return render_template('Days_on_market.html')

@app.route('/predict_DOM', methods=['POST'])
def predict_DOM():
    #int_features = [float(x) for x in request.form.values()]
    #Change based on model working
    zip_code = int(request.form['zip_code'])
    central_air = int(request.form['central_air'])
    garage_spaces = float(request.form['garage_spaces'])
    number_of_bedrooms = float(request.form['number_of_bedrooms'])
    number_of_bathrooms = float(request.form['number_of_bathrooms'])
    number_stories = float(request.form['number_stories'])
    total_livable_area = float(request.form['total_livable_area'])
    total_area = float(request.form['total_area'])
    year_built = int(request.form['year_built'])
    has_basements = int(request.form['has_basements'])
    sale_price = int(request.form['sale_price'])

    #zip_code = int_features[0]

    filepath = 'data\\Zip_Reference.xlsx'
    df = pd.read_excel(filepath)

    #df = df[df['zip_code'] == zip_code]

    zip_row = df[df['zip_code'] == zip_code]
    if zip_row.empty:
        return render_template('index.html', prediction_text='Invalid Zip Code. Please try again.')
    
    house_features = zip_row.drop(columns=['zip_code']).values.flatten().tolist()

    #Combining Excel Values with Given Values
    final_features = [[
            zip_code, central_air, garage_spaces, number_of_bedrooms,
            number_of_bathrooms, number_stories, total_livable_area,
            total_area, year_built, has_basements,
            sale_price, central_air] + house_features]
    
    #Predict based on values
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('Days_on_market.html', prediction_text=f'The time on market for the house is {output}')



if __name__ == "__main__":
    app.run(debug=True)
