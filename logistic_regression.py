import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_confusion_matrix, plot_roc_curve

from matplotlib import pyplot as plt 

# Set up own cleaned_data_full

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
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
    plot_roc_curve(model, X_test, y_test)                     
    
    return model, accuracy, roc_auc, time_taken

df = pd.read_csv('data/cleaned_comments.csv')
X = df.loc[:, df.columns != "label"]
x_train, x_test, y_train, y_test = train_test_split(X, df['label'], random_state=42)

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
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3), min_df=0.01, max_df=0.9)
    vectorizer.fit(x_train['comment'])
    x_train = vectorizer.transform(x_train['comment'])
    x_test = vectorizer.transform(x_test['comment'])

    
    model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, x_train, y_train, x_test, y_test)
    return model_lr, accuracy_lr, roc_auc_lr, tt_lr

#logit_tfidf("data/cleaned_comments.csv")
# Scalar features logit
### Normalize Features
from sklearn.preprocessing import StandardScaler
def normalize_scalar(x_train, x_test):
    scalar_features = ['score', 'ups', 'downs']
    scaler = StandardScaler()
    x_train.loc[:, scalar_features] = scaler.fit_transform(x_train[scalar_features])
    x_test.loc[:, scalar_features] = scaler.fit_transform(x_test[scalar_features])
    return x_train.loc[:, scalar_features], x_test.loc[:, scalar_features]

def logit_scalar(csv):
    # Split data
    df = pd.read_csv(csv)
    X = df.loc[:, df.columns != "label"]
    x_train, x_test, y_train, y_test = train_test_split(X, df['label'], random_state=42)
    
    # Model - standard
    params_lr = {'penalty': 'elasticnet', 'l1_ratio':0.5, 'solver': 'saga'}
    model_lr = LogisticRegression(**params_lr)
    
    # Normalize scalar
    x_train, x_test = normalize_scalar(x_train, x_test)
    model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, x_train, y_train, x_test, y_test)
    return model_lr, accuracy_lr, roc_auc_lr, tt_lr

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
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3), min_df=0.008, max_df=0.8)
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
    
        
    model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, train, y_train, test, y_test)


print("COMBINEd")
logit_tfidf("data/cleaned_comments.csv")
print("SCALAR")
logit_scalar("data/cleaned_comments.csv")
print("TF_IDF")
logit_combined("data/cleaned_comments.csv")


#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cv_train = x_train.copy()
cv_train['comment'] =cv.fit_transform(x_train['comment']).todense()
cv_test =cv.transform(x_test['comment'])

#model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, cv_train, y_train, cv_test, y_test)

# Feature selection
# from sklearn.feature_selection import SelectKBest, chi2
# modified_data = pd.DataFrame(r_scaler.transform(train), columns=train.columns)
# X = modified_data.loc[:,modified_data.columns!='label']
# y = modified_data[['label']]
# selector = SelectKBest(chi2, k=10)
# selector.fit(X, y)
# X_new = selector.transform(X)
# print(X.columns[selector.get_support(indices=True)])