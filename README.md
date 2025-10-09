# Titanic Survival Prediction

This project aims to predict the survival of passengers aboard the Titanic using machine learning models. 


## Project Overview

The project involves the following steps:
1. Exploratory Data Analysis (EDA) to understand the dataset.
2. Data preprocessing and cleaning.
3. Training multiple machine learning models.
4. Evaluating model performance.
5. Predicting survival on new data.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repo url>
   cd titanic_survival_prediction
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

---
## Models
The following machine learning models are trained and saved in the models/ directory:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Kernel SVM
Each model is saved as a `.joblib` file and can be loaded for predictions.

---

## Usage
### Training Models
To train a specific model, run the corresponding script in the `scripts/` directory. For example:

```bash
python -m scripts.train_logistic_regression
```

During training, key metrics such as accuracy, precision, recall, and F1-score are printed to the console to help evaluate the model's performance.

---

### Predicting Survival
To predict survival using a trained model, use the `predict_survival.py` script:

```bash
python -m scripts.predict_survival
```

**Note:** Before running the prediction script, ensure that the path to the trained model file is correctly set inside the `predict_survival.py` script.  
Update the `model_path` variable to point to the desired model file in the `models/` directory.

---
## Dataset
The dataset used is sourced from Kaggle: [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data).



