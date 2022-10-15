import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def svm(csv):
    df = pd.read_csv(csv)
    # remove non english comments?

    # all columns except "labels" column
    X = df.loc[:, df.columns != "label"]

    # (true labels) labels: 1 = sarcastic, 0 = not sarcastic
    y = df["label"].to_numpy()

    # split the dataset into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    C = 1.0 # SVM regularization parameter
    # svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y)
    LinearSVM = LinearSVC().fit(X_train, y_train)
    print("training set score: %f" % LinearSVM.score(X_train, y_train))
    print("test set score: %f" % LinearSVM.score(X_test, y_test))
    

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