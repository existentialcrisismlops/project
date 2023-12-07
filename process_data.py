import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path='dummy_sensor_data.csv', test_size=0.2, random_state=42):
    # Load the data from DVC-managed file
    data = pd.read_csv(file_path)

    # Display a sample of the raw data
    print("Raw Data:")
    print(data.head())

    # Extract features and target variable
    X = data[['Machine_ID', 'Sensor_ID', 'Reading']]
    
    # Convert timestamps to numerical values (e.g., seconds since a reference point)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Timestamp'] = (data['Timestamp'] - data['Timestamp'].min()).dt.total_seconds()
    y = data['Timestamp']

    # Convert categorical variables into numerical representation (if needed)
    X = pd.get_dummies(X)

    # Normalize or scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # Display the processed data
    print("\nProcessed Data:")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)

    return X_train, X_val, y_train, y_val, scaler

# If you want to run this script independently
if __name__ == "__main__":
    X_train, X_val, y_train, y_val, scaler = preprocess_data()
