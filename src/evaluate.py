import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model():
    df = pd.read_csv('data/raw/train.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Unseen raw test data (keeping 'date' column intact)
    test_data = df.iloc[-180:].copy()
    
    X_test = test_data.drop('sales', axis=1)
    y_test = test_data['sales']
    
    # Load trained pipeline
    pipeline = joblib.load('models/pipeline.pkl')
    
    # Direct inference: Pipeline will automatically extract features and predict!
    predictions = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print("=== Model Evaluation Results ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

if __name__ == '__main__':
    evaluate_model()
