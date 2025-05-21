<h1 align='center'>Prediction Website</h1>

![Website](https://github.com/user-attachments/assets/b67495fb-bf0c-47ab-9ff4-ef76d5537154)



------------------------
DESCRIPTION
------------------------
This project is a web-based machine learning application that predicts either house sale prices or time on market using home attributes and locations (ZIP code). 

- 'FinalEval Sale Model Code.ipynb': This Jupyter notebook is used to build the learning model that predicts the estimated sale price of a home. When preparing for training, split the dataset into into 70% training set and a 30% test set. Select key variables using Elastic Net. Train the training set using linear regression, random forest, and gradient boosting, evaluating their performance using RMSE and R2. Using RandomizedSearchCV, optimize parameters for gradient boosting. The best performing model from the  randomized search is then exported as 'modelSale.pkl'.
- 'FinalEval TOM Model Code.ipynb': This Jupyter notebook is used to build the learning model that predicts the time-on-market of a home. When preparing for training, split the dataset into into 70% training set and a 30% test set. Select key variables using Elastic Net. Train the training set using linear regression, random forest, and gradient boosting, evaluating their performance using RMSE and R2. Using RandomizedSearchCV, optimize parameters for gradient boosting. The best-performing model from the randomized search is then exported as 'modelTOM.pkl'.

Within the 'Web site' folder:
- 'app.Combined.py': The main Flask web application. This script loads the pre-trained models, displays an HTML page, processes user inputs, and returns predictions for the home price and time-on-market.

Within the 'Web site\data' folder:
- 'Zip_Reference.xlsx': A reference dataset containing ZIP-code-specific features. These features are automatically appended to user inputs to enhance the model's prediction.

Within the 'Web site\models' folder:
- 'modelSale.pkl': Pre-trained model that predicts the estimated sale price of a home.
- 'modelTOM.pkl':  Pre-trained model that estimates the time-on-market of a home.

Within the 'Web site\template' folder:
- 'index.html': The HTML template containing a form where the user inputs property details to predict the estimated sale price.
- 'Days_on_market.html': The HTML template containing a form where users input property details to predict the estimated time-on-market.
    Both pages use Flask, integrating user input, ZIP code data, and pre-trained models to return predictions.

Within the 'DataProcessing' folder:
- 'CleaningPhillyData.ipynb': This Jupyter notebook is used to gather data from "https://cityofphiladelphia.github.io/carto-api-explorer/#opa_properties_public". Uses SQL queries to read in the data. Keeps relevant features that users can input into the prediction model. Generate flags for missing data to help improve the prediction model accuracy. Uses Histogram-based Gradient Boosting Imputation to fill missing numeric values. Transforms features into numeric scales to allow use in learning models. Reorganizes and exports the dataset into a structured format suitable for machine learning models.
- 'CleaningRedFinDataset.ipynb': This Jupyter notebook is used to gather data from "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz". Reads in data from the download link. Creates a  ZIP code column, then keeps relevant features that can impact the length of a home on the market. Gather recent crime data from "https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads". Reformat crime data into state abbreviations to attach to Redfin's dataset by state. Group columns by zipcode and property type to provide missing data and rename columns. Select features to keep by reorganizing  into a structured format suitable to append to Philly's dataset.
- 'PhillyAndRedFin.ipynb': This Jupyter notebook is used to combine our Philly dataset and Redfin dataset. Merge data based on the zipcode and type of property using a scale to help map the property by total area. Check for missing data and fill in any missing data with a fallback. Generate an estimated listing date using the average days on market and sale date. Turn dates into integers to allow use in machine learning models. Uploads the final training and testing dataset.


------------------------
INSTALLATION
------------------------
    Installing required packages
1. Download and extract the project files.
2. Open a terminal and navigate into the project folder. 
3. Install required packages with "pip install -r requirements.txt" to install all dependencies.

    Download Datasets From Scratch
1. Open Data Processing Folder
2. Run CleaningPhillyData.ipynb to get data from Philly
3. Run CleaningRedFinDataset.ipynb to get data from RedFin
4. Run PhillyAndRedFin.ipynb to combine RedFin and Philly 

------------------------
EXECUTION
------------------------
1. In the terminal, navigate into the 'Website' folder
2. Run "python appCombined.py" to start up the Flask web app.
3. Once Flask is running, open up a browser and go to "http://127.0.0.1:5000/"
4. On the homepage, you can switch between predicting the listed price for a home or click the green button to estimate a home's time-on-market.
5. Enter Zthe IP code and property features and submit the form. The predicted sale price or estimated time-on-market will be returned at the top of the page.
- You may regenerate the learning models or modify them inside "FinalEval Sale Model Code.ipynb" and "FinalEval TOM Model Code.ipynb".
