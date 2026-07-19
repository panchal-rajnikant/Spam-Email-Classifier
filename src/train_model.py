from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate

def train(X, y):
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    lr_cross = cross_validate(lr_pipe, X, y, cv=5, scoring=scoring)
    nb_cross = cross_validate(nb_pipe, X, y, cv=5, scoring=scoring)
    return lr_pipe, nb_pipe, lr_cross, nb_cross 

  
