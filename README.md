# NER_CRF

Jupyter Notebook NER_CRF_GIT 

import pandas as pd
import numpy as np

#data = pd.read_csv("C:/Users/samsung/Downloads/ner_dataset.csv", encoding="latin1")
data = pd.read_csv("C:/Users/samsung/Downloads/Book1.csv", encoding = "latin1")
import pandas as pd
import numpy as np
​
#data = pd.read_csv("C:/Users/samsung/Downloads/ner_dataset.csv", encoding="latin1")
data = pd.read_csv("C:/Users/samsung/Downloads/Book1.csv", encoding = "latin1")

data = data.fillna(method="ffill")
#data.head()

words = list(set(data["Word"].values))

n_words = len(words) 
n_words
n_words = len(words) 
n_words

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)

sent = getter.get_next()

print(sent)

sentences = getter.sentences

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
​
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
​
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
​
    return features
​
​
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
​
def sent2labels(sent):
    return [label for token, postag, label in sent]
​
def sent2tokens(sent):
    return [token for token, postag, label in sent]

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

from sklearn_crfsuite import CRF
​
crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

#import nltk
#import sklearn_crfsuite

#!pip install sklearn_crfsuite

from sklearn_crfsuite import CRF
crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=2)

pred

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X, y)

import eli5

eli5.show_weights(crf, top=30)
