# Car-Price-Prediction
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

file_path = "car-data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found.")

df = pd.read_csv(file_path)
df['Car_Age'] = datetime.now().year - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
df.fillna(0, inplace=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_cols)],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
