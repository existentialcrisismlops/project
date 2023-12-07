import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data from DVC-managed file
data_file_path = 'dummy_sensor_data.csv'
data = pd.read_csv(data_file_path)

# Display a sample of the raw data
print("Raw Data:")
print(data.head())

# Extract features and target variable
X = data[['Machine_ID', 'Sensor_ID', 'Reading']]
y = data['Timestamp']

# Convert categorical variables into numerical representation (if needed)
X = pd.get_dummies(X)

# Normalize or scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Display the processed data
print("\nProcessed Data:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
