import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import gensim
from gensim.models import Word2Vec

from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

def svm(csv): # best so far
    df = pd.read_csv(csv)

    # all columns except "labels" column
    X = df.loc[:, df.columns != "label"]
    x_comment = X.comment

    y = df["label"]

    vect = CountVectorizer(strip_accents='unicode', max_df=0.70)
    tf_trans = TfidfTransformer()

    # create the SVM model
    svm_model = SGDClassifier(penalty="elasticnet", random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ('vect', vect),
        ('tftrans', tf_trans),
        ('model', svm_model)
    ])
    
    param_grid = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_features': (5000, 15000, 30000),
        'model__l1_ratio': (0.0, 0.15, 0.40, 0.60, 0.85, 1.0)
    }

    grid_svm = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=3, n_jobs=-1)
    grid_svm.fit(x_comment, y)

    print(grid_svm.best_score_)
    for param_name in sorted(param_grid.keys()):
        print("%s: %r" % (param_name, grid_svm.best_params_[param_name]))


def svm2(csv):
    df = pd.read_csv(csv)

    x = df.comment
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    count_vect = CountVectorizer(strip_accents='unicode', max_df=0.70)
    c = count_vect.fit(X_train)
    vect = TfidfTransformer()

    count_X_train = c.transform(X_train)
    count_X_test = c.transform(X_test)
    
    vec_X_train = vect.transform(count_X_train)
    vec_X_test = vect.transform(count_X_test)

    svc = LinearSVC(penalty='l1', C=0.55, fit_intercept=False, dual=False, tol=1e-10, max_iter=100000)

    svc.fit(vec_X_train, y_train)
    y_pred = svc.predict(vec_X_test)
    print(classification_report(y_pred, y_test))

def svm3(csv): # using word2vec / sent2vec
    df = pd.read_csv(csv)

    # all columns except "labels" column
    # X = df.loc[:, df.columns != "label"]

    x = df.comment
    y = df.label

    # TODO: implement
    model = gensim.models.Word2Vec(x,
                 vector_size=100
                 # Size is the length of our vector.
                )


    # X_train, X_test, y_train, y_test = train_test_split(x, y)


def main(args):
    assert args.csv, "Please specify --csv"
    svm(args.csv)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='path to the dataset file')
    return parser.parse_args()

# Usage: python svm.py --csv data/cleaned_comments.csv
if __name__ == "__main__":
    args = get_arguments()
    main(args)