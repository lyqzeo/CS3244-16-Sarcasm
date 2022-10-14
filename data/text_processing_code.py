import pandas as pd
import numpy as np
import string
import nltk
nltk.download("all")

import pickle as pk

import warnings
warnings.filterwarnings("ignore")

import urllib

from bs4 import BeautifulSoup
import unicodedata
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords

from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk import ne_chunk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

###### PLEASE SET YOUR OWN LOCAL PATH ######
local_path = ".../train-balanced-sarcasm.csv" 

## Reading CSV
df = pd.read_csv(local_path)
df.head()

## Removing Empty Rows
#print(sum(df["comment"].isnull()))   # Before
nan_value = float("NaN") #Convert NaN values to empty string
df.replace("", nan_value, inplace=True)
df.dropna(subset = ["comment"], inplace=True)
#print(sum(df["comment"].isnull()))  # After

## Checking for duplicates
#df.duplicated(subset = ['comment', 'parent_comment']).value_counts() # Before
df = df.drop_duplicates(subset=['comment','parent_comment'], keep='first', inplace=False, ignore_index=False)
#df.duplicated(subset = ['comment', 'parent_comment']).value_counts() # After

## Converting pandas columns to appropriate types
df['comment'] = df['comment'].astype(str)
df['author'] = df['author'].astype(str)
df['subreddit'] = df['subreddit'].astype(str)
df['parent_comment'] = df['parent_comment'].astype(str)
df['date'] = df['date'].astype('datetime64[ns]')
df['created_utc'] = df['created_utc'].astype('datetime64[ns]')

## Checking for data imbalance
#df['label'].value_counts()
def clean_text(df, category):
    all_reviews = list()
    lines = df[category].values.tolist()
    for text in lines:
        text = text.lower()
        
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        
        emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        text = emoji.sub(r'', text)
        
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)        
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text) 
        text = re.sub(r"\'ll", " will", text)  
        text = re.sub(r"\'ve", " have", text)  
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"did't", "did not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"have't", "have not", text)

        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        tokens = word_tokenize(text)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]   # Stemming
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        words = [w for w in words if not w in stop_words]
        words = ' '.join(words)
        all_reviews.append(words)
    return all_reviews

cleaned_comments = clean_text(df, "comment")
cleaned_parent_comment = clean_text(df, "parent_comment")

# cleaned_comments and cleaned_parent_comment not in df
