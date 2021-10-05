import nltk
from nltk.stem.lancaster import LancasterStemmer
from tflearn.layers.core import activation
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as f:
    data = json.load(f)


words = []
labels = []
docs_x = []
docs_y = []

for intents in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x,doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

output_row = out_empty[:]
output_row[labels.index(docs_y[x])] = 1


training.append(bag)
output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()
#input data
net = tflearn.input_data(shape=[None, len(training[0])])
#8 neurons for the hidden layer 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#output layer
net = tflearn.fully_connected(net, len(output[0], activation="softmax"))
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")