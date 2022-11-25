import argparse
#from tkinter.ttk import _TreeviewTagDict

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
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.sparse import csr_matrix
from mlxtend.preprocessing import DenseTransformer


def svm(csv): # best so far, cross validation
    df = pd.read_csv(csv)

    # all columns except "labels" column
    X = df.loc[:, df.columns != "label"]
    x_comment = X.comment
    x_comment = X.comment.astype(str)

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
    
    # param_grid = {
    #     'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #     'vect__max_features': (5000, 15000, 30000),
    #     'model__l1_ratio': (0.0, 0.15, 0.40, 0.60, 0.85, 1.0)
    # }   
    # param_grid = {
    #     'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1,4)],
    #     'vect__max_features': (5000, 15000, 30000),
    #     'model__l1_ratio': (0.0, 0.15, 0.40, 0.60, 0.85, 1.0)
    # }   

    param_grid = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3),(1,4), (1,5), (1,6), (1,7), (1,8), (1,9)],
        'vect__max_features': (5000, 10000, 15000, 20000, 30000),
        'model__l1_ratio': (0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1,0)
    }   


    grid_svm = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=3, n_jobs=-1)
    grid_svm.fit(x_comment, y)

    print("tfidf + gridsearch")
    print(grid_svm.best_score_)
    for param_name in sorted(param_grid.keys()):
        print("%s: %r" % (param_name, grid_svm.best_params_[param_name]))
    


def svmtfidf(csv):    #Countvect + tfidf + svm
    df = pd.read_csv(csv)

    x = df.comment
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    count_vect = CountVectorizer(strip_accents='unicode', max_df=0.70)
    c = count_vect.fit_transform(X_train)
    vect = TfidfTransformer(smooth_idf=True,use_idf=True)
    vect.fit(c)

    count_X_test = count_vect.transform(X_test)
    vec_X_test = vect.transform(count_X_test)

    svc = LinearSVC(penalty='l1', C=0.55, fit_intercept=False, dual=False, tol=1e-10, max_iter=100000)

    svc.fit(c, y_train)
    y_pred = svc.predict(vec_X_test)
    print("tfidf + defaultsvm")
    print(classification_report(y_pred, y_test))
    

def svmvader(csv): # using vader
    df = pd.read_csv(csv)

    x = df.vader_comment
    y = df.label

    x_scores = np.array([eval(i)['compound'] for i in x])
    x_scores = x_scores.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(x_scores, y)

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    
    svc = LinearSVC(penalty='l1', C=0.55, fit_intercept=False, dual=False, tol=1e-10, max_iter=100000)

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print("vader + default")
    print(classification_report(y_pred, y_test))

## Accuracy 
# Compound = 0.50-0.51
# Pos = 0.47
# Neg = 0.54
# Neu = 0.48
# Pos-Neg = 0.49
# Pos+Neg = 0.55


def svmlda(csv): ## Using lda
    
    df = pd.read_csv(csv)

    x = df.comment
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    count_vect = CountVectorizer(strip_accents='unicode', max_df=0.70)
    c = count_vect.fit_transform(X_train)
    vect = TfidfTransformer(smooth_idf=True,use_idf=True)
    vect_X_train = vect.fit_transform(c).todense()

    count_X_test = count_vect.transform(X_test)
    vec_X_test = vect.transform(count_X_test).todense()

    #pca = PCA(n_components = 500)
    #pca_xtrain = pca.fit_transform(c) ## fits and transforms
    #pca_xtest = pca.transform(vec_X_test) ## transforms maps fitted para 
    
    lda = LDA(n_components = 1)
    X_train_lda = lda.fit_transform(vect_X_train, y_train)
    X_test_lda = lda.transform(vec_X_test)


    svc = LinearSVC(penalty='l1', C=0.55, fit_intercept=False, dual=False, tol=1e-10, max_iter=100000)

    svc.fit(X_train_lda, y_train)
    y_pred = svc.predict(X_test_lda)

    print("tfidf + lda + default svm")
    print(classification_report(y_pred, y_test))

def svmd(csv): #tfidf + lda + gridsvm
    df = pd.read_csv(csv)

    # all columns except "labels" column
    X = df.loc[:, df.columns != "label"]
    x_comment = X.comment
    x_comment = X.comment.astype(str)

    y = df["label"]

    vect = CountVectorizer(strip_accents='unicode', max_df=0.70)
    tf_trans = TfidfTransformer()

    svm_model = SGDClassifier(penalty="elasticnet", random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ('vect', vect),
        ('tftrans', tf_trans),
        ('to_dense', DenseTransformer()),
        ('lda', LDA(n_components = 1)),
        ('model', svm_model)
    ])
    
    param_grid = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3),(1,4), (1,5), (1,6), (1,7), (1,8), (1,9)],
        'vect__max_features': (5000, 10000, 15000, 20000, 30000),
        'model__l1_ratio': (0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1,0)
    }   

    grid_svm = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=3, n_jobs=-1)
    grid_svm.fit(x_comment, y)

    print("tfidf + lda + gridsearch")
    print(grid_svm.best_score_)
    for param_name in sorted(param_grid.keys()):
        print("%s: %r" % (param_name, grid_svm.best_params_[param_name]))

def svmwrd(csv):

    df = pd.read_csv(csv)

    x = df.comment.astype(str)
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    words = []
    for i in X_train.values:
        words.append(i.split())

    model = gensim.models.Word2Vec(words, vector_size=150, window = 15, min_count = 1, epochs=15)

    words = set(model.wv.index_to_key )
    X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words])
                            for ls in X_train], dtype = object)
    X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words])
                            for ls in X_test], dtype= object)

    # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(150, dtype=float))
            
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(150, dtype=float))


    svc = LinearSVC(penalty='l2', C=0.55, fit_intercept=False, dual=False, tol=1e-10, max_iter=100000)
    svc.fit(X_train_vect_avg, y_train)

    y_pred = svc.predict(X_test_vect_avg)

    print("word2vec + default")
    print(classification_report(y_pred, y_test))


    

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