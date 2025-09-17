import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Initialize the app
app = Flask(__name__)

# Load the encoder, scaler, and model
encoder_path = 'models/encoder.pkl'
scaler_path = 'models/scaler.pkl'
model_path = 'models/logistic_regression_best_model.pkl'

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': float(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        # Prepare dataframe
        import pandas as pd
        df = pd.DataFrame([input_data])

        # Encode categorical variables
        categorical_cols = encoder.feature_names_in_
        encoded_features = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )

        # Drop and merge
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        # Scale numerical columns
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[scale_cols] = scaler.transform(df[scale_cols])

        # Predict
        prediction = model.predict(df)
        pred_label = 'Yes' if int(prediction[0]) == 1 else 'No'

        # 🔴 Changed to result.html
        return render_template('result.html', prediction=f"Churn Prediction: {pred_label}")

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
