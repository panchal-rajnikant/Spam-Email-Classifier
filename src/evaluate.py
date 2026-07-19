import pandas as pd

def compare_models(lr_cross, nb_cross):

    comparison = pd.DataFrame({
        "Model":[
            "Logistic Regression",
            "Naive Bayes"
        ],
        "Accuracy":[
            lr_cross["test_accuracy"].mean(),
            nb_cross["test_accuracy"].mean()
        ],
        "Precision":[
            lr_cross["test_precision"].mean(),
            nb_cross["test_precision"].mean()
        ],
        "Recall":[
            lr_cross["test_recall"].mean(),
            nb_cross["test_recall"].mean()
        ],
        "F1 Score":[
            lr_cross["test_f1"].mean(),
            nb_cross["test_f1"].mean()
        ],
        "ROC-AUC":[
            lr_cross["test_roc_auc"].mean(),
            nb_cross["test_roc_auc"].mean()
        ],
    })

    print(comparison)