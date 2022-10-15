# Possible guide: https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d

import argparse

import numpy as np
import pandas as pd

def cnn(csv):
    df = pd.read_csv(csv)
    #TODO
    pass

def main(args):
    assert args.csv, "Please specify --csv"
    cnn(args.csv)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='path to the dataset file')
    return parser.parse_args()

# Usage: python svm.py --csv data/cleaned_comments.csv
if __name__ == "__main__":
    args = get_arguments()
    main(args)