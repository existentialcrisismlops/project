from flask import Flask, render_template
import mlflow.sklearn
import pandas as pd
import mlflow
import os
from process_data import preprocess_data

app = Flask(__name__)


# Specify the MLflow experiment name and run ID
mlflow_experiment_name = "wistful-wren-342"
run_id = "5e4f59df99d742218fc5c9ae617bd397"
artifact_uri = f"file:///proj/mlruns/0/{run_id}/artifacts/model"

loaded_model = mlflow.sklearn.load_model(artifact_uri)

@app.route('/predict')
def predict():
    # Get the input data (replace this with your actual input data)
    path = 'dummy_sensor_data.csv'
    x_val,_,_,_,_ = preprocess_data(path)
    predictions = loaded_model.predict(x_val)

    # Log the predictions as a metric (replace this with your actual metrics)
    mlflow.log_metric("prediction_mean", predictions.mean())

    return f"Predictions: {predictions.tolist()}"

if __name__ == "__main__":
    app.run(debug=True, port=8080,host='0.0.0.0')