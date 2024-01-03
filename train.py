import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import sys

def train_model(X_train, y_train, X_val, y_val, n_estimators, max_depth, random_state):
    with mlflow.start_run():

        # Enable automatic logging for scikit-learn
        mlflow.sklearn.autolog()

        # Create and train the model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val)

        # Calculate the mean squared error
        mse = mean_squared_error(y_val, y_val_pred)

        # Note: You don't need explicit calls to mlflow.log_params and mlflow.sklearn.log_model

        # No need to save the model explicitly; autolog() handles it automatically

        return model

if __name__ == "__main__":
    # Load preprocessed data
    from process_data import preprocess_data
    X_train, X_val, y_train, y_val, _ = preprocess_data()

    # Get parameters from command line
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    random_state = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    # Train the model with hyperparameter tuning (RandomizedSearchCV example)
    param_distributions = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15]
    }
    model = RandomizedSearchCV(RandomForestRegressor(random_state=random_state),
                               param_distributions=param_distributions,
                               n_iter=3,  
                               scoring='neg_mean_squared_error',
                               cv=3,
                               verbose=1,
                               n_jobs=-1)
    trained_model = train_model(X_train, y_train, X_val, y_val, n_estimators, max_depth, random_state)
    # Register the best model in MLflow Model Registry
    best_run = mlflow.search_runs(order_by=["metrics.neg_mean_squared_error"]).iloc[0]
    best_run_id = best_run.run_id
    mlflow.register_model(f"runs:/{best_run_id}/best_model", "Best Model")
