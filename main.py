import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from numpy import argmax
import textblob
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer


#csv from kugle
data = pd.read_csv(r'blogtext.csv',nrows=10000)

raw_text=data["text"]




#remove all bad chars
data["text"]=data["text"].replace('[^\\w\\s]','',regex=True)

#stopwords
stop = stopwords.words('english')
data["text"] = data["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#more frequented words
all_words = ' '.join(raw_text).split()
freq = pd.Series(all_words).value_counts()[:20]

data["text"] = data["text"].apply(lambda t: ' '.join(word for word in t.split() if word not in freq))
#remove numbers
data["text"]=data["text"].replace('[0-9]','',regex=True)

#lemmatization
lemmatizer = WordNetLemmatizer()
data["text"] = data["text"].apply(lambda t: ' '.join([lemmatizer.lemmatize(word) for word in t.split()]))




print(data["text"].head(50))
f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["text"])
data["gender"]=data["gender"].map({'male':1,'female':0})
print(data["gender"].head(50))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['gender'], test_size=0.33, random_state=42)



# #Naive Bayes
list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1

matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(20))

best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]

best_index = models[models['Test Precision']>0.50]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
models.iloc[best_index, :]

m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print("Naive Bayes")
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
                   index = ['Actual 0', 'Actual 1']))


# SVM
#
# svc = svm.SVC(C=10000,gamma='auto')
# svc.fit(X_train, y_train)
# m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
# print("SVM")
# print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
#                    index = ['Actual 0', 'Actual 1']))
