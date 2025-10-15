import pandas as pd
import numpy as np
from joblib import load
from src.data_cleaning import clean_data
import os

# load trained model
model_path = './models/model_random_forest.joblib'  
pipeline = load(model_path)

# load new data 
new_data = pd.read_csv('./data/titanic_dataset_new.csv')

# Clean the data
X,y = clean_data(new_data)

# predict
y_pred = pipeline.predict(X)

# file path
pred_path = './results/predict_survival.csv'

pred_df = pd.DataFrame({
    'Survived': y,
    'random_forest_pred': y_pred
})

if not os.path.exists(pred_path):
    pred_df.to_csv(pred_path, index=False)
    print(f"Created new file: {pred_path}")
else:
    existing = pd.read_csv(pred_path)
    if len(existing) == len(y):
        existing['random_forest_pred'] = y_pred
        existing.to_csv(pred_path, index=False)
        print(f"Updated existing file: {pred_path}")
    else:
        print("Warning: row counts differ — not updating the file.")