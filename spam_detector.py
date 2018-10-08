# spam detector using the SMS Spam Collection Dataset by UCI Machine Learning
# https://www.kaggle.com/uciml/sms-spam-collection-dataset

# data analysis and wrangling
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

# machine learning
from sklearn.naive_bayes import MultinomialNB

# metrics 
from sklearn import metrics

spam = pd.read_csv("./spam.csv", encoding = "ISO-8859-1")

X = spam.v2
y = spam.v1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

vectorizer = CountVectorizer()
vectorizer.stop_words = text.ENGLISH_STOP_WORDS
X_train = vectorizer.fit_transform(X_train)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

def test():
    y_pred = mnb.predict(vectorizer.transform(X_test))
    print("Multinomial Naive Bayes on Spam SMS Dataset Accuracy (in %):", metrics.accuracy_score(y_test, y_pred)*100)
    
def predict(text):
    if not (isinstance(text, list)):
        text = [text]
    pred = mnb.predict(vectorizer.transform(text))
    print(pred)