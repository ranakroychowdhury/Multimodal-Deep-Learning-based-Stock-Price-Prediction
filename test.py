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

from keras.models import Model, load_model
from keras.layers import Dense, concatenate, Input, TimeDistributed, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from Lib.config import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import sys
import os
import shutil
import json

DATA_DIR = 'Data/'
TMP_DIR = 'Tmp/'


def test(x_news_test, x_price_test, y_test, model):
	price_scl = joblib.load(TMP_DIR + 'price_scl.scl')
	label_scl = joblib.load(TMP_DIR + 'label_scl.scl')

	y_predict = model.predict([x_price_test] + x_news_test)

	yy_test = label_scl.inverse_transform(y_test)
	yy_predict = label_scl.inverse_transform(y_predict)

	error = np.abs(yy_test - yy_predict)
	error = np.divide(error, yy_test)
	error = np.abs(np.mean(error, axis=0))

	print("Error per company: " + str(error))
	print("Total error: " + str(np.mean(error)))

	return np.mean(error)


if __name__ == '__main__':
	TEST_DATA_PATH = sys.argv[1]
	TEST_DATA_DIR = os.path.dirname(TEST_DATA_PATH) + '/'
	shutil.unpack_archive(TEST_DATA_PATH, extract_dir=TEST_DATA_DIR)

	x_price_test = np.load(TEST_DATA_DIR + 'x_price_test.npy')
	x_news_test = list(np.load(TEST_DATA_DIR + 'x_news_test.npy'))
	y_test = np.load(TEST_DATA_DIR + 'y_test.npy')

	MODEL_FILE = sys.argv[2]
	
	model = load_model(MODEL_FILE)

	error = test(x_news_test, x_price_test, y_test, model)
	print('Test Error:', error)

	os.remove(TEST_DATA_DIR + 'x_price_test.npy')
	os.remove(TEST_DATA_DIR + 'x_news_test.npy')
	os.remove(TEST_DATA_DIR + 'y_test.npy')



