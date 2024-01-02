import mlflow
from mlflow import pyfunc
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
# Import the preprocess_data function from process_data script
from process_data import preprocess_data

def fetch_model(run_id):
    """
    Fetches a model from MLflow using the given run ID.
    """
    # Update the path to the local directory where the model artifacts are stored
    local_model_path = f"file:///C:/Users/User/Desktop/semester7/MLOPs/project/project/mlruns/0/{run_id}/artifacts/model"
    model = pyfunc.load_model(local_model_path)
    return model

def fetch_data(data_file_path):
    """
    Fetches and preprocesses new data for prediction.
    """
    # Use the preprocess_data function to preprocess the data
    X_train, X_val, y_train, y_val, scaler = preprocess_data(file_path=data_file_path, test_size=0.2, random_state=42)
    return X_val, y_val

def save_metrics(mae, mse, r2, filename='metrics.csv'):
    """
    Saves the calculated metrics to a CSV file.
    """
    metrics = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'R2': [r2]})

    # Check if the file exists
    if os.path.exists(filename):
        existing_metrics = pd.read_csv(filename)
        updated_metrics = pd.concat([existing_metrics, metrics], ignore_index=True)
        updated_metrics.to_csv(filename, index=False)
    else:
        metrics.to_csv(filename, index=False)


def monitor_model(run_id, data_file_path, threshold_mae=16, threshold_mse=500):
    """
    Monitors the model for concept drift by evaluating its performance on new data.
    """
    model = fetch_model(run_id)
    X, y = fetch_data(data_file_path)

    predictions = model.predict(X)

    # Calculate error metrics
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    save_metrics(mae, mse, r2)

    # Check for concept drift
    if mae > 16 or mse > 500:
        print("Concept drift detected.")
        return True
    else:
        print("No concept drift detected.")
        return False

if __name__ == '__main__':
    run_id = "8554082c3edd4d0b831c49077368690d"  # Your model's run ID from meta.yaml
    data_file_path = "dummy_sensor_data.csv"  # Path to your new data
    monitor_model(run_id, data_file_path)
