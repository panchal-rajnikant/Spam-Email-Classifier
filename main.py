from src import preprocess, train_model, evaluate
import pandas as pd
import joblib

df = pd.read_csv("data/spam.csv")

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["message"] = df["message"].apply(preprocess.clean_text)

X = df["message"]
y = df["label"]

lr_pipe, nb_pipe, lr_cross, nb_cross = train_model.train(X, y)

evaluate.compare_models(lr_cross, nb_cross)

# Save the best model 

lr_f1 = lr_cross["test_f1"].mean()
nb_f1 = nb_cross["test_f1"].mean()

if lr_f1 >= nb_f1:
    best_model = lr_pipe
    best_model_name = "Logistic Regression"
else:
    best_model = nb_pipe
    best_model_name = "Multinomial Naive Bayes"

joblib.dump(best_model, "models/spam_model.pkl")

print(f"{best_model_name} saved successfully.")