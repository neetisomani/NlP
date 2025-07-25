import numpy as np
import pandas as pd
dataset = pd.read_csv('./data/a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)
dataset.shape
dataset.head()
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
corpus=[]

for i in range(0, 900):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
corpus
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)
# Saving BoW dictionary to later use in prediction
import pickle
bow_path = './data/c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Exporting NB Classifier to later use in prediction
import joblib
joblib.dump(classifier, './data/c2_Classifier_Sentiment_Model') 
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)


**************************************************************************

import numpy as np
import pandas as pd
dataset = pd.read_csv('./data/testing data/IMDB.csv')
dataset.head()
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
corpus=[]

for i in range(0, 200):
  review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
# Loading BoW dictionary
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cvFile='./data/model/c1_BoW_Sentiment_Model.pkl'
# cv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./drive/MyDrive/Colab Notebooks/2 Sentiment Analysis (Basic)/3.1 BoW_Sentiment Model.pkl', "rb")))
cv = pickle.load(open(cvFile, "rb"))
X_fresh = cv.transform(corpus).toarray()
X_fresh.shape
X_fresh
import joblib
classifier = joblib.load('./data/model/c2_Classifier_Sentiment_Model')
y_pred = classifier.predict(X_fresh)
print(y_pred)
# dataset['predicted_label'] = y_pred.tolist()
dataset['predicted_label'] = pd.Series(y_pred)

dataset.head()
dataset.to_csv("./data/predicated data/c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)