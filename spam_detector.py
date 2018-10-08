# spam detector using the SMS Spam Collection Dataset by UCI Machine Learning
# https://www.kaggle.com/uciml/sms-spam-collection-dataset

import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

spam = pd.read_csv("./spam.csv", encoding = "ISO-8859-1")

X = spam.v2
y = spam.v1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

def test():
    from sklearn import metrics
    y_pred = mnb.predict(vectorizer.transform(X_test))
    print("Multinomial Naive Bayes on Spam SMS Dataset Accuracy (in %):", metrics.accuracy_score(y_test, y_pred)*100)
    
def predict(text):
    if not (isinstance(text, list)):
        text = [text]
    pred = mnb.predict(vectorizer.transform(text))
    print(pred)