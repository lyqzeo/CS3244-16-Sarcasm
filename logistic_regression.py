import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_confusion_matrix, plot_roc_curve

from matplotlib import pyplot as plt 

def run_model(model, X_train, y_train, X_test, y_test, name, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.pink, normalize = 'all')
    plot_roc_curve(model, X_test, y_test, name=name)                     
    
    return model, accuracy, roc_auc, time_taken

# Remove stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
stop_words.append("would")
def removeStopWords(text):
    # Remove stop words
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(token)
    return " ".join(tokens)

# Basic tf-idf
def logit_tfidf(csv):
    # split data and remove stopwords
    df = pd.read_csv(csv)
    X = df.loc[:, df.columns != "label"]
    x_train, x_test, y_train, y_test = train_test_split(X, df['label'], random_state=42)
    x_train.comment = x_train.comment.apply(lambda x: removeStopWords(x))
    
    # Model - standard
    params_lr = {'penalty': 'elasticnet', 'l1_ratio':0.5, 'solver': 'saga'}
    model_lr = LogisticRegression(**params_lr)
    
    # Tf-idf (fit on x_train)
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2))
    vectorizer.fit(x_train['comment'])
    x_train = vectorizer.transform(x_train['comment'])
    x_test = vectorizer.transform(x_test['comment'])

    
    model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, x_train, y_train, x_test, y_test, "tf-idf")
    return model_lr, accuracy_lr, roc_auc_lr, tt_lr

# Scalar features logit
### Normalize Features
from sklearn.preprocessing import StandardScaler
def normalize_scalar(x_train, x_test):
    scalar_features = ['score', 'ups', 'downs']
    scaler = StandardScaler()
    x_train.loc[:, scalar_features] = scaler.fit_transform(x_train[scalar_features])
    x_test.loc[:, scalar_features] = scaler.fit_transform(x_test[scalar_features])
    return x_train.loc[:, scalar_features], x_test.loc[:, scalar_features]

# tf-idf combined
def logit_combined(csv):
    # split data and remove stopwords
    df = pd.read_csv(csv)
    X = df.loc[:, df.columns != "label"]
    x_train, x_test, y_train, y_test = train_test_split(X, df['label'], random_state=42)
    x_train.comment = x_train.comment.apply(lambda x: removeStopWords(x))
        
    # Model - standard
    params_lr = {'penalty': 'elasticnet', 'l1_ratio':0.5, 'solver': 'saga'}
    model_lr = LogisticRegression(**params_lr)
        
    # Tf-idf (fit on x_train)
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2))
    vectorizer.fit(x_train['comment'])
    
    tf_idf_train = pd.DataFrame(vectorizer.transform(x_train['comment']).toarray(), 
            columns=vectorizer.get_feature_names())
    tf_idf_test = pd.DataFrame(vectorizer.transform(x_test['comment']).toarray(), 
            columns=vectorizer.get_feature_names())
    
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    
    scalar_features = ['score', 'ups', 'downs']
    train = pd.concat([x_train.loc[:, scalar_features], tf_idf_train], axis=1)
    test = pd.concat([x_test.loc[:, scalar_features], tf_idf_test], axis=1)
    
        
    model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, train, y_train, test, y_test, "Combined")

def logit_tfidf_limited(csv):
    # split data and remove stopwords
    df = pd.read_csv(csv)
    X = df.loc[:, df.columns != "label"]
    x_train, x_test, y_train, y_test = train_test_split(X, df['label'], random_state=42)
    x_train.comment = x_train.comment.apply(lambda x: removeStopWords(x))
    
    # Model - standard
    params_lr = {'solver': 'liblinear'}
    model_lr = LogisticRegression(**params_lr)
    
    # Tf-idf (fit on x_train)
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2), min_df=0.008, max_df=0.8)
    vectorizer.fit(x_train['comment'])
    x_train = vectorizer.transform(x_train['comment'])
    x_test = vectorizer.transform(x_test['comment'])

    
    model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, x_train, y_train, x_test, y_test, "tf-idf limited")
    return model_lr, accuracy_lr, roc_auc_lr, tt_lr

print("TF_IDF")
logit_tfidf("data/cleaned_comments.csv")
# print("TF_IDF_lcombined")
# logit_combined("data/cleaned_comments.csv")
print("TF_IDF_limited")
logit_tfidf_limited("data/cleaned_comments.csv")