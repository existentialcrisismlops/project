import mlflow.sklearn
import pandas as pd

def load_live_data():
    # Implement code to load live data
    live_data = pd.read_csv('dummy_sensor_data.csv')  # Load live data from a CSV file
    return live_data

def preprocess_live_data(data):
    # Verify the column names using live_data.columns
    print(data.columns)  # Print column names to check if 'Timestamp' exists

    # Replace 'Timestamp' with the actual column name from your live data
    # Check the column names and use the correct one
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        data['Timestamp'] = pd.to_numeric(data['Timestamp'], errors='coerce')

    # Convert specific columns to numeric format, excluding non-numeric columns
    numeric_columns = ['Reading']  # Add other columns that should be numeric
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')

    return data

def test_model_on_live_data():
    run_id = "8554082c3edd4d0b831c49077368690d"  # Replace <RUN_ID> with the actual run ID
    model_uri = f"runs:/{run_id}/model"  # Create the correct URI
    model = mlflow.sklearn.load_model(model_uri)

    live_data = load_live_data()
    live_data = preprocess_live_data(live_data)  # Preprocess live data

    predictions = model.predict(live_data)

    # Perform further operations with predictions or results as needed
    # Example:
    # predictions.to_csv('predicted_results.csv', index=False)

if __name__ == "__main__":
    test_model_on_live_data()
