import joblib
def predict_email(emails):

    model = joblib.load("models/lr_spam_model.pkl")
    # prediction = model.predict([emails])[0]

    # return "Spam" if prediction == 1 else "Ham"

    predictions = model.predict(emails)

    for email, pred in zip(emails, predictions):
        label = "Spam" if pred == 1 else "Ham"
        print(f"{email} --> {label}")



