import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('Dataset2'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import spacy
import csv


def read_data(path):
    with open(path, 'r', encoding="mbcs") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        labels = []
        sentences = []
        for row in readCSV:
            label = row[0]
            sentence = row[1]
            labels.append(label)
            sentences.append(sentence)
    return sentences, labels



sentences_test,labels_test = read_data('Dataset2/test.csv')
print(sentences_test[:3],'\n')
print(labels_test[:3])


sentences_train,labels_train = read_data('Dataset2/train.csv')


import spacy
import numpy as np


nlp = spacy.load('en_vectors_web_lg')

embedding_dim = nlp.vocab.vectors_length

print(embedding_dim)

def encode_sentences(sentences):

    n_sentences = len(sentences)

    print('Length :-',n_sentences)

    X = np.zeros((n_sentences, embedding_dim))
    #y = np.zeros((n_sentences, embedding_dim))


    for idx, sentence in enumerate(sentences):

        doc = nlp(sentence)
        X[idx, :] = doc.vector
    return X

train_X = encode_sentences(sentences_train)
test_X = encode_sentences(sentences_test)

def label_encoding(labels):


    n_labels = len(labels)
    print('Number of labels :-',n_labels)



    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y =le.fit_transform(labels)
    print(y[:100])
    print('Length of y :- ',y.shape)
    return y

train_y = label_encoding(labels_train)
test_y = label_encoding(labels_test)


df1 = pd.read_csv('Dataset2/train.csv', delimiter=',')
df1.dataframeName = 'train.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

print(df1.sample(10))

df1.describe()


import matplotlib.pyplot as plt
import seaborn as sns


plt.hist(train_y)


plt.title('Histogram of Intent Lables')
plt.xlabel('Intent Types')
plt.ylabel('Frequency')
plt.show()



from sklearn.svm import SVC


def svc_training(X,y):
    clf = SVC(C=1)
    clf.fit(X, y)
    return clf

model = svc_training(train_X,train_y)




def svc_validation(model,X,y):

    y_pred = model.predict(X)


    n_correct = 0
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            n_correct += 1

    print("Predicted {0} correctly out of {1} training examples".format(n_correct, len(y)))


svc_validation(model,train_X,train_y)
svc_validation(model,test_X,test_y)

from sklearn.metrics import classification_report
y_true, y_pred = test_y, model.predict(test_X)
print(classification_report(y_true, y_pred))