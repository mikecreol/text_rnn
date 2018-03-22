
#https://www.youtube.com/watch?v=DDByc9LyMV8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import imdb

# Data Import
#------------------------------------------------------------------------------

imdb.maybe_download_and_extract()

x_train_text, y_train = imdb.load_data(train=True) # I added utf-8 encoding in the code in imdb
x_test_text, y_test = imdb.load_data(train=False)

data_text = x_train_test + x_test_text

# Tokenizer
#------------------------------------------------------------------------------

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data_text)

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

#x_train_text[3]
#x_train_tokens[3][0:10]
#
## here in some examples the first word is not found because num_words is 10000 and some words are skipped
#list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(11)]

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

19:10