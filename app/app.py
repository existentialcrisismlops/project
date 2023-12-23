from flask import Flask
import mlflow.sklearn
import pandas as pd
import mlflow

# Set the correct path to the mlruns directory on your machine
mlflow.set_tracking_uri("file:///D:/Main/Semesters/7TH/mlops_work/project/project/mlruns/models")

app = Flask(__name__)

# Specify the MLflow experiment name and run ID
mlflow_experiment_name = "shivering-carp-573"
run_id = "8554082c3edd4d0b831c49077368690d"

# Load the MLflow model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/modal1")

# Get the path to the loaded model
model_path = mlflow.sklearn.get_model_path(model)

print("Model path:", model_path)

@app.route('/predict')
def predict():
    # Get the input data (replace this with your actual input data)
    data = {"feature1": 1, "feature2": 2, "feature3": 3}
    df = pd.DataFrame([data])

    # Make predictions using the loaded model
    predictions = model.predict(df)

    # Log the predictions as a metric (replace this with your actual metrics)
    mlflow.log_metric("prediction_mean", predictions.mean())

    return f"Predictions: {predictions.tolist()}"

if __name__ == "__main__":
    app.run(debug=True)
