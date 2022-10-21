import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv('data/cleaned_comments.csv')

logit = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.11)

vectorizer = TfidfVectorizer(use_idf=True, min_df=2, ngram_range=(1,1))
X_tf_idf= vectorizer.fit_transform(train_df['comment'])
x_train, x_test, y_train, y_test = train_test_split(X_tf_idf, train_df['label'], random_state=42)

logit.fit(x_train, y_train)
print(accuracy_score(y_test, logit.predict(x_test)))
#0.66

report = classification_report(logit.predict(x_test), y_test, output_dict=True)
report = pd.DataFrame(report).transpose()
report = report.rename({"0": "Sarcastic Comments", "1": "Non Sarcastic Comments"})
print(report)
