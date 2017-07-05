import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
import re
import json

# I've used this tutorial: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/.
# I've used this code on FloydHub

def readfile(fname):
    with open(fname, 'r') as f:
        return f.read().lower()

def preprocess(poem):
    # left only words
    poem = re.sub('[^!а-яіїєА-ЯІЇЄ\s\,\.\-\—\:\n\!\(\)\?’]', ' ', poem)
    return poem.replace('\t', '\n')

folder = '/input/'
# I've decided to use only one book to train
file = 'Stus_Vasyl.Tom_3_1.Palimpsesty.1576.ua.txt' #list(filter(lambda x: 'Palimpsesty' in x, os.listdir(folder)))[0]
raw_text = preprocess(readfile(folder + file))
chars = set(raw_text)
char_to_int =  { c:i for i, c in enumerate(chars) }
int_to_char = {i:c for i, c in enumerate(chars)}
with open('/output/char_to_int.json', 'w') as f:
    json.dump(char_to_int, f)
with open('/output/int_to_char.json', 'w') as f:
    json.dump(int_to_char, f)

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: {}".format(n_chars))
print("Total Vocab: {}".format(n_vocab))

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: {}".format(n_patterns))

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="/output/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)