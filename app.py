from flask import Flask
import mlflow.sklearn
import pandas as pd
import mlflow
from process_data import preprocess_data

# Set the correct path to the mlruns directory on your machine
mlflow.set_tracking_uri("file:///D:/Main/Semesters/7TH/mlops_work/project/project/mlruns")

app = Flask(__name__)

# Specify the MLflow experiment name and run ID
mlflow_experiment_name = "wistful-wren-342"
run_id = "5e4f59df99d742218fc5c9ae617bd397"

# Load the MLflow model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

@app.route('/predict')
def predict():
    # Get the input data (replace this with your actual input data)
    path = 'dummy_sensor_data.csv'
    x_val,_,_,_,_ = preprocess_data(path)
    predictions = model.predict(x_val)

    # Log the predictions as a metric (replace this with your actual metrics)
    mlflow.log_metric("prediction_mean", predictions.mean())

    return f"Predictions: {predictions.tolist()}"

if __name__ == "__main__":
    app.run(debug=True)
