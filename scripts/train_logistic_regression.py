import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import full_preprocessor
from src.data_cleaning import clean_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# load data
df = pd.read_csv("./data/titanic_dataset.csv")

# Clean the data
X,y = clean_data(df)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# build full model pipeline 
pipeline = Pipeline(steps=[
    ('preprocess', full_preprocessor),
    ('model', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))
])

# train
pipeline.fit(X_train, y_train)

# evaluate model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"f1_score: {f1:.3f}")


# Save trained pipeline
dump(pipeline, './models/model_logistic_regression.joblib')
print("Saved model to './models/model_logistic_regression.joblib'")