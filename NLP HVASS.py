
#https://www.youtube.com/watch?v=DDByc9LyMV8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import timeit
import ipdb

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, auc

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import imdb
import helper_functions

# Data Import
#------------------------------------------------------------------------------

imdb.maybe_download_and_extract()

x_train_text, y_train = imdb.load_data(train=True) # I added utf-8 encoding in the code in imdb
x_test_text, y_test = imdb.load_data(train=False)

data_text = x_train_text + x_test_text

# Tokenizer
#------------------------------------------------------------------------------

num_words = 30000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data_text)

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

#x_train_text[3]
#x_train_tokens[3][0:10]
#
## here in some examples the first word is not found because num_words is 10000 and some words are skipped
#list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(11)]

# inverse mapping from a list of tokens to text
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))
def tokens_to_string(tokens):
    ipdb.set_trace()
    words = [inverse_map[token] for token in tokens if token != 0] # zero is used for padding
    text = " ".join(words)
    
    return text

# Padding
#------------------------------------------------------------------------------

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

#np.sum(num_tokens < max_tokens) / len(num_tokens) # hpw much of the data will be truncated
#np.percentile(num_tokens, 99)
#np.max(num_tokens)
#plt.hist(num_tokens, bins=15)

# it is good to pad the zeros to the beginning
# it is better to truncate in batches not the whole thing as you are loosing less data. Padding is not good 
#as a whole and the less you do it the better

pad = "pre"
x_train_pad = pad_sequences(x_train_tokens, 
                            maxlen = max_tokens,
                            padding = pad,
                            truncating = pad)

x_test_pad = pad_sequences(x_test_tokens, 
                           maxlen = max_tokens,
                           padding = pad,
                           truncating = pad)

# Constructing the model in keras; Embedding will be a part of the model
#------------------------------------------------------------------------------

model = Sequential()

embedding_size = 8 # the lenghts of the vectors after the embedding

# Embedding
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name="layer_embedding"))

# Recurrent units
model.add(GRU(units=16, return_sequences=True)) # LSTMs might be worse as they have redundent gates
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4)) # this one outputs only the final output

# fully connected layer
model.add(Dense(1, activation='sigmoid'))

#------------------------------------------------------------------------------

optimizer = Adam(lr=1e-3)

model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

model.summary()

# training of the model
start = timeit.default_timer()
model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=64)
stop = timeit.default_timer()
stop - start

# Performance evaluation
#------------------------------------------------------------------------------

start = timeit.default_timer()
model.evaluate(x_test_pad, y_test)
stop = timeit.default_timer()
stop - start

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]
cls_pred = np.array([1.0 if p>0.5 else 0 for p in y_pred])
cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)

idx = incorrect[0]

confusion_matrix(cls_pred, cls_true)

y_pred1 = model.predict(x=x_test_pad)
y_pred2 = np.array([1.0 if p>0.5 else 0 for p in y_pred1])
confusion_matrix(y_pred2, y_test)
np.mean(y_pred2 == y_test)

Gini(y_test, y_pred1) # GINI index messes up in the checks cause some of the vectors are 1 dimensional

new_tokens = tokenizer.texts_to_sequences(["this moovie is really bad"])

tokens_to_string(new_tokens)


27:45