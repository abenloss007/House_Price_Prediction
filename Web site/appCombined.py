import numpy as np 
from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__, template_folder='template')

# Load models
model_price = joblib.load('models/modelSale.pkl')
model_tom = joblib.load('models/modelTOM.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/days_on_market')
def days_on_market():
    return render_template('Days_on_market.html')

#Predict Price
@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
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
        days_on_market = int(request.form['days_on_market'])

        filepath = os.path.join('data', 'Zip_Reference.xlsx')
        df = pd.read_excel(filepath)

        #Creating fallback zipcode if unavailable
        columns = df.columns

        default_zip_row = [zip_code, 63.5267404, 65.37853275,50.60538082, 47.43445478, 0.964606059, 0.173866758, 5461.711538, 0.257197022, 188.2471329, 257567.819, 171.5007468, 238710.8712]

        default_zip_row_df = pd.DataFrame([default_zip_row], columns=columns)

        if zip_code in df['zip_code'].values:
            zip_row = df[df['zip_code'] == zip_code]
        else:
            zip_row = default_zip_row_df

        house_features = zip_row.drop(columns=['zip_code']).values.flatten().tolist()

        final_features = [[
            zip_code, central_air, garage_spaces, number_of_bedrooms,
            number_of_bathrooms, number_stories, total_livable_area,
            total_area, year_built, has_basements,
            days_on_market,
        ] + house_features]

        prediction = model_price.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'The predicted price of the house is ${output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

#Predict Time on Market
@app.route('/predict_dom', methods=['POST'])
def predict_dom():
    try:
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

        filepath = os.path.join('data', 'Zip_Reference.xlsx')
        df = pd.read_excel(filepath)
        
        #Creating fallback zipcode if unavailable
        columns = df.columns

        default_zip_row = [zip_code, 63.5267404, 65.37853275,50.60538082, 47.43445478, 0.964606059, 0.173866758, 5461.711538, 0.257197022, 188.2471329, 257567.819, 171.5007468, 238710.8712]

        default_zip_row_df = pd.DataFrame([default_zip_row], columns=columns)

        if zip_code in df['zip_code'].values:
            zip_row = df[df['zip_code'] == zip_code]
        else:
            zip_row = default_zip_row_df

        house_features = zip_row.drop(columns=['zip_code']).values.flatten().tolist()

        final_features = [[
            zip_code, central_air, garage_spaces, number_of_bedrooms,
            number_of_bathrooms, number_stories, total_livable_area,
            total_area, year_built, has_basements,
            sale_price,
        ] + house_features]

        prediction = model_tom.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('Days_on_market.html', prediction_text=f'The estimated time on market is {output} days')

    except Exception as e:
        return render_template('Days_on_market.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
