import argparse
import re

import contractions
import pandas as pd
import numpy as np
import string
import nltk
# nltk.download("all")

import warnings
warnings.filterwarnings("ignore")

from nltk import pos_tag
from nltk import ne_chunk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def preprocess(input_path, output_file):
    ## Reading CSV
    df = pd.read_csv(input_path)

    ## Checking for duplicates
    df = df.drop_duplicates(subset=['comment','parent_comment'])

    ## Converting pandas columns to appropriate types
    df['comment'] = df['comment'].astype(str)
    df['author'] = df['author'].astype(str)
    df['subreddit'] = df['subreddit'].astype(str)
    df['parent_comment'] = df['parent_comment'].astype(str)
    df['date'] = df['date'].astype('datetime64[ns]')
    df['created_utc'] = df['created_utc'].astype('datetime64[ns]')

    ## Checking for data imbalance
    # df['label'].value_counts()

    preprocess_cat(df, "comment")
    preprocess_cat(df, "parent_comment")

    ## Removing Empty Rows
    nan_value = float("NaN") # Convert NaN values to empty string
    df.replace("", nan_value, inplace=True)
    df.dropna(subset=["comment"], inplace=True)

    # cleaned_comments and cleaned_parent_comment not in df
    df.to_csv(output_file, index=False)

def preprocess_cat(df, category):
    # lowercase all comments
    df[category] = df[category].str.lower()

    # remove urls
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    df[category] = np.vectorize(pattern.sub)('', df[category])

    emoji = re.compile("["
                u"\U0001F600-\U0001FFFF"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
    df[category] = np.vectorize(emoji.sub)(r'', df[category])

    # expand contractions ie. he's => he is
    df[category] = np.vectorize(contractions.fix)(df[category])

    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")

    for i in df[category].index:
        text = df.at[i, "comment"]

        tokens = word_tokenize(text)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens] # stemming
        words = [word for word in stripped if word.isalpha()]

        words = [w for w in words if not w in stop_words]
        words = ' '.join(words)
        df.at[i, "comment"] = words

def main(args):
    assert args.input and args.output, "Please specify --input and --output"
    preprocess(args.input, args.output)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='file to be pre-processed')
    parser.add_argument('--output', help='pre-processed file')
    return parser.parse_args()

# example:
# python text_processing_code.py --input train-subset.csv --output cleaned_comments.csv
if __name__ == "__main__":
    args = get_arguments()
    main(args)
