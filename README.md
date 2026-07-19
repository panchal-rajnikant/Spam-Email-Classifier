# Spam-Email-Classifier

A machine learning project that classifies emails as Spam or Ham using Natural Language Processing (NLP).  The project compares Logistic Regression and Multinomial Naive Bayes using TF-IDF features.

## Features

- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Model Comparison
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- 5-Fold Cross Validation
- Save Best Model

## Technologies

- Python
- Pandas
- Scikit-learn
- Joblib

## Project Structure

spam-email-classifier/

├── data/
├── models/
├── src/
├── README.md
├── requirements.txt
└── main.py

## Installation
pip install -r requirements.txt

## Run
python main.py

## Workflow

Raw Emails
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train Models
      │
      ▼
Evaluation
      │
      ▼
Best Model Saved

## Results

Example:

      Model             Accuracy  Precision  Recall  F1 Score  ROC-AUC
Logistic Regression       0.8        0.7     0.8  0.733333      1.0
Naive Bayes               0.8        0.7     0.8  0.733333      1.0


## Future Improvements

- FastAPI REST API
- Web Interface
- Larger Dataset
- Docker Support
- Deploy on Render or Railway

