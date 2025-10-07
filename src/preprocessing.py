import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# custom transformers
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_['Title'] = X_['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        X_['Deck'] = X_['Cabin'].astype(str).str[0].replace('n', 'U')
        X_['IsAlone'] = ((X_['SibSp'] + X_['Parch'] + 1) == 1).astype(int)
        X_ = X_.drop(columns=['Name', 'Cabin', 'SibSp', 'Parch'])
        return X_
    
# feature definitions
log_feature = ['Fare']
non_log_feature= ['Age']
categorical_features = ['Embarked', 'Deck', 'Sex', 'Title']

def make_numeric_transformer(scale=True):
    norm_steps= [('impute', SimpleImputer(strategy='mean'))]
    log_steps = [
        ('impute', SimpleImputer(strategy='mean')),
        ('log', FunctionTransformer(np.log1p, validate=False))
    ]

    if scale:
        log_steps.append(('scaler', StandardScaler()))
        norm_steps.append(('scaler', StandardScaler()))
    
    log_numeric_transformer = Pipeline(steps=log_steps)
    normal_numeric_transformer = Pipeline(steps=norm_steps)

    numeric_transformer = ColumnTransformer(transformers=[
        ('log_num', log_numeric_transformer, log_feature),
        ('norm_num', normal_numeric_transformer, non_log_feature)
    ]) 

    return numeric_transformer

def build_preprocessor(scale=True):
    numeric_transformer = make_numeric_transformer(scale=scale)
    categorical_transformer = OneHotEncoder(
        drop='first', handle_unknown='ignore', sparse_output=False
    )

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, log_feature + non_log_feature),
        ('cat', categorical_transformer, categorical_features)
    ])

    full_pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    return full_pipeline


full_preprocessor = build_preprocessor(scale=True)  
full_preprocessor_no_scale = build_preprocessor(scale=False)
    



