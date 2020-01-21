import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer

from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import Model
from keras.layers import Dense, concatenate, Input, TimeDistributed, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

import sys
import os
import shutil
import json

from Lib.config import *


DATA_DIR = 'Data/'


def get_model(lookback, price_length, news_length, news_embedding_dim, max_words, nb_labels, optim, lstm_unit, lossfn):
	price_input = Input(shape=(lookback, price_length), name='price_input')
	news_inputs = [Input(shape=(news_length, ), name='news_lookback_' + str(i + 1)) for i in range(lookback)]
	news_word_embedding = Embedding(max_words, news_embedding_dim)
	news_words_embeddings = [news_word_embedding(inp) for inp in news_inputs]
	concatenated_news = concatenate(news_words_embeddings, axis=1)
	reshaped_news = Reshape((lookback, news_length * news_embedding_dim))(concatenated_news)
	concatenated_all = concatenate([price_input, reshaped_news], axis=-1)
	lstm = LSTM(lstm_unit)(concatenated_all)
	output = Dense(nb_labels)(lstm)

	model = Model(inputs=[price_input]+news_inputs, outputs=[output])
	model.compile(optimizer=optim, loss=lossfn)
	return model


def train(x_news_train, x_price_train, y_train, optim, lossfn, epochs, lstm_unit):
	model = get_model(lookback, price_length, news_length, 
		news_embedding_dim, max_words, nb_labels, optim, lstm_unit, lossfn)
	h = model.fit([x_price_train] + x_news_train, y_train, 
		epochs=epochs,
		shuffle=False)

	return model





if __name__ == '__main__':
	TRAIN_DATA_PATH = sys.argv[1]
	TRAIN_DATA_DIR = os.path.dirname(TRAIN_DATA_PATH) + '/'
	shutil.unpack_archive(TRAIN_DATA_PATH, extract_dir=TRAIN_DATA_DIR)

	x_price_train = np.load(TRAIN_DATA_DIR + 'x_price_train.npy')
	x_news_train = list(np.load(TRAIN_DATA_DIR + 'x_news_train.npy'))
	y_train = np.load(TRAIN_DATA_DIR + 'y_train.npy')


	HYPER_FILE = sys.argv[2]
	best_hyp = json.load(open(HYPER_FILE))

	optim = best_hyp['optim']
	epochs = best_hyp['epochs']
	lossfn = best_hyp['lossfn']
	lstm_unit = best_hyp['lstm_unit']

	model = train(x_news_train, x_price_train, y_train, optim, lossfn, epochs, lstm_unit)
	model.save('model.h5')

	os.remove(TRAIN_DATA_DIR + 'x_price_train.npy')
	os.remove(TRAIN_DATA_DIR + 'x_news_train.npy')
	os.remove(TRAIN_DATA_DIR + 'y_train.npy')
