import pandas as pd
import joblib
from preprocessing import get_full_pipeline

def train_complete_model():
    # Load raw data
    df = pd.read_csv('data/raw/train.csv')
    
    # Optional: memory optimization
    df['store'] = df['store'].astype('int8')
    df['item'] = df['item'].astype('int16')
    df['sales'] = df['sales'].astype('float32')
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Taking data from 2016 onwards to save memory
    df = df[df['date'].dt.year >= 2016]
    
    # Split train data (keeping 'date' column so FunctionTransformer can use it!)
    train_data = df.iloc[:-180]
    
    X_train = train_data.drop('sales', axis=1)
    y_train = train_data['sales']
    
    # Get the pipeline containing FunctionTransformer
    pipeline = get_full_pipeline()
    
    print("Training the full pipeline (including automated feature engineering)...")
    pipeline.fit(X_train, y_train)
    print("Training completed successfully!")
    
    joblib.dump(pipeline, 'models/pipeline.pkl')
    print("Trained pipeline saved to models/pipeline.pkl")

if __name__ == '__main__':
    train_complete_model()
