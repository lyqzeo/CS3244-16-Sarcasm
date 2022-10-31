# Possible guide: https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d
# Note: Might change from cnn to regular nn for ease of implementation
#       But might use cnn anyway for potentially powerful model

import os

import argparse

import numpy as np
import pandas as pd
import gensim.downloader
import json

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# Code inspired by: https://towardsdatascience.com/super-easy-way-to-get-sentence-embedding-using-fasttext-in-python-a70f34ac5b7c
wordEmbeddingsModel = None
def loadWordEmbeddingsModel(): # Defer Expensive load
    global wordEmbeddingsModel
    wordEmbeddingsModel = gensim.downloader.load('fasttext-wiki-news-subwords-300')

def sentenceVec(sentence):
    i = 0
    acc = np.zeros(300)
    if type(sentence) is float: #Handle edge cases where sentence is nan (Not a Number), due to weird empty string logic
        return acc #Return 0 vector as edge case
    for word in sentence.split():
        i+=1
        if word in wordEmbeddingsModel: #NOTE: Hack to work around out of vocab feature not working
            #NOTE: Fasttext *should* work with out of vocab words, but this model doesn't automatically
            acc += wordEmbeddingsModel[word] #TODO: Make this work with out of vocab
    return acc/i #TODO: Test this

def extractVaderList(str):
    well_formatted_str = str.replace("'",'"')
    vader_dict = json.loads(well_formatted_str)
    return list(vader_dict[k] for k in ('neg', 'neu', 'pos', 'compound'))

def get_X_y(csv):
    if os.path.isfile("./data/cnn_cache.npz"):
        print("Reading from cached X,y")
        arr = np.load("data/cnn_cache.npz")
        return arr["X"], arr["y"]
    
    print("No cached X,y found, computing from source")

    loadWordEmbeddingsModel()

    df = pd.read_csv(csv)
    comments_vader = df.apply(
        lambda row: extractVaderList(row.vader_comment), 
        axis=1, 
        result_type='expand'
      ).rename(columns={0:'neg', 1:'neu', 2:'pos', 3:'compound'})
    comments_embed = df.apply(lambda row: sentenceVec(row.comment), axis=1, result_type='expand')
    #TODO: See about adding sentence embbed and vader for parent and post
    X = pd.concat([comments_embed, comments_vader], axis=1).to_numpy()
    y = df["label"].to_numpy()
    np.savez("data/cnn_cache", X=X, y=y)
    return X,y

def cnn(csv):
    X,y = get_X_y(csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #TODO: experiment with hyperparameters
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(30, 5), random_state=1)
    clf = clf.fit(X_train, y_train)

    print("training set score: %f" % clf.score(X_train, y_train))
    print("test set score: %f" % clf.score(X_test, y_test))

def main(args):
    assert args.csv, "Please specify --csv"
    cnn(args.csv)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='path to the dataset file')
    return parser.parse_args()

# Usage: python cnn.py --csv data/cleaned_comments_full.csv
if __name__ == "__main__":
    args = get_arguments()
    main(args)