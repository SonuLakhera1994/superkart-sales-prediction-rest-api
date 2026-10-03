
# Import necessary libraries
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
product_Store_Sales_api = Flask("SuperKart product store sales Predictor")

# Load the trained machine learning model
model = joblib.load("super_kart_prediction_model_v2_0.joblib")


# Define a route for the home page
@product_Store_Sales_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


# Define an endpoint for single prediction
@product_Store_Sales_api.post('/v1/predict')
def predict_sales():

    # Get JSON data from request body
    property_data = request.get_json()

    # Extract relevant features
    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_MRP': property_data['Product_MRP'],
        'Store_Age_Years': property_data['Store_Age_Years'],
        'Product_Sugar_Content': property_data['Product_Sugar_Content'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type'],
        'Product_Id_char': property_data['Product_Id_char'],
        'Product_Type_Category': property_data['Product_Type_Category']
    }

    # Convert input into DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert NumPy value to Python float and round
    predicted_sales = round(float(predicted_sales), 2)

    # Return prediction
    return jsonify({
        'Predicted Sales': predicted_sales
    })


# Define an endpoint for batch prediction
@product_Store_Sales_api.post('/v1/predictbatch')
def predict_sales_batch():

    # Get uploaded CSV file
    file = request.files['file']

    # Read CSV
    input_data = pd.read_csv(file)

    # Make predictions
    predicted_sales = model.predict(input_data)

    # Convert predictions to Python floats
    predicted_sales = [
        round(float(prediction), 2)
        for prediction in predicted_sales
    ]

    # Create dictionary using IDs
    property_ids = input_data['id'].tolist()

    output_dict = dict(zip(property_ids, predicted_sales))

    # Return predictions
    return output_dict


# Run Flask application
if __name__ == '__main__':
    product_Store_Sales_api.run(debug=True)
