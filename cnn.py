# Possible guide: https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d
# Note: Might change from cnn to regular nn for ease of implementation
#       But might use cnn anyway for potentially powerful model

import os, time

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

    ### CONFIG ###
    X_input_to_include = []
    X_input_to_include.append("comments_vader")
    X_input_to_include.append("comments_embed")
    X_input_to_include.append("p_comments_vader")
    X_input_to_include.append("p_comments_embed")
    print("Logging data config:", X_input_to_include)

    X_input = []

    if ("comments_embed" in X_input_to_include) or ("p_comments_embed" in X_input_to_include):
        print("Loading Word Embedding Model")
        loadWordEmbeddingsModel()
        print("Finished Loading Model")
    else:
        print("Skip loading Word Embedding Model")

    df = pd.read_csv(csv)

    if "comments_vader" in X_input_to_include:
        comments_vader = df.apply(
            lambda row: extractVaderList(row.vader_comment), 
            axis=1, 
            result_type='expand'
        ).rename(columns={0:'neg', 1:'neu', 2:'pos', 3:'compound'})
        X_input.append(comments_vader)

    if "comments_embed" in X_input_to_include:
        comments_embed = df.apply(
            lambda row: sentenceVec(row.comment), axis=1, result_type='expand'
            ).rename(lambda x : str(x), axis='columns')
        X_input.append(comments_embed)
    
    if "p_comments_vader" in X_input_to_include:
        p_comments_vader = df.apply(
            lambda row: extractVaderList(row.vader_pcomment), 
            axis=1, 
            result_type='expand'
        ).rename(columns={0:'p_neg', 1:'p_neu', 2:'p_pos', 3:'p_compound'})
        X_input.append(p_comments_vader)
    
    if "p_comments_embed" in X_input_to_include:
        p_comments_embed = df.apply(
            lambda row: sentenceVec(row.parent_comment), axis=1, result_type='expand'
            ).rename(lambda x : "p_"+str(x), axis='columns')
        X_input.append(p_comments_embed)
    
    X = pd.concat(X_input, axis=1).to_numpy()
    y = df["label"].to_numpy()

    np.savez("data/cnn_cache", X=X, y=y)
    
    return X,y

def cnn(csv):
    X,y = get_X_y(csv)
    print("Data shape: ", X.shape,)
    print("Loading complete, setting up...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #TODO: experiment with hyperparameters
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(30, 5), random_state=1, verbose=True)
    print("Logging hyper-parameters")
    print("Solver:", clf.__dict__["solver"])
    print("Alpha:", clf.__dict__["alpha"])
    print("Hidden layer sizes", clf.__dict__["hidden_layer_sizes"])
    print("")

    print("Begin training...")
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
    start_time = time.time()
    args = get_arguments()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))