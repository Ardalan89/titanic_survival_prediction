import numpy as np
import pandas as pd


def clean_data(df):
    
    df = df.copy()
    
    # keep required columns
    req_cols = ['Name','Sex', 'Age','SibSp', 'Parch', 'Cabin', 'Fare', 'Embarked', 'Survived']
    df = df[req_cols]

    # handling Embarked missing values
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # handling Fare missing values
    df.loc[df['Fare']==0, 'Fare'] = np.nan

    # separate target and features
    y = df['Survived']
    X = df.drop(columns=['Survived'])

    return X,y