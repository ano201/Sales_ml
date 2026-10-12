import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def extract_time_features(df):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year.astype('int16')
        df['month'] = df['date'].dt.month.astype('int8')
        df['day'] = df['date'].dt.day.astype('int8')
        df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')
        df = df.drop('date', axis=1)
    return df

def get_full_pipeline():
    pipeline = Pipeline([
        ('feature_engineering', FunctionTransformer(extract_time_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42, n_estimators=10, max_depth=10, n_jobs=-1))
    ])
    return pipeline
