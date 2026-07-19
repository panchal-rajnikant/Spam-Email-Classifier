from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, train_test_split
import joblib

def train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    # train logistic regression
    lr_pipe.fit(X_train, y_train)
    # save logistic regression model
    joblib.dump(lr_pipe, "models/lr_spam_model.pkl")


    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    lr_cross = cross_validate(lr_pipe, X, y, cv=5, scoring=scoring)
    nb_cross = cross_validate(nb_pipe, X, y, cv=5, scoring=scoring)
    return lr_pipe, nb_pipe, lr_cross, nb_cross 

  
