import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
import shutil
import os
from os import path
import sys

from Lib.config import *


DATA_DIR = 'Data/'
TEMP_DATA_DIR= 'Tmp'
DATA_PATH = sys.argv[1]

if not os.path.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)

shutil.unpack_archive(DATA_PATH, extract_dir=DATA_DIR)
data = pd.read_csv(DATA_DIR + 'All_new.csv')


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,2:7]))

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(headlines)


# In[10]:

def news_to_vec(data, start_index, end_index):
    pads = []
    for i in range(start_index, end_index+1):
        sequences_data = tokenizer.texts_to_sequences(data.iloc[:, i])
        padded_data_seqs = sequence.pad_sequences(sequences_data, maxlen=per_news_length)
        pads.append(padded_data_seqs)
    return np.hstack(pads)


def lookback_price(data, lb):
    X = []
    for i in range(len(data) - lb - 1):
        X.append(data[i:(i+lb), :])
    return np.array(X)


# In[5]:

def lookback_news(data, lb):
    X = []
    for i in range(lb):
        X.append([])
    for i in range(len(data) - lb - 1):
        for j in range(0, lb):
            X[j].append(data[i + j, :])
    for i in range(lb):
        X[i] = np.array(X[i])
    return X


# In[6]:

def lookback_label(data, lb):
    X = []
    for i in range(len(data) - lb - 1):
        X.append(data[i+lb, :])
    return np.array(X)


# In[11]:

news_data = news_to_vec(data, 2, 6)
x_news = lookback_news(news_data, lookback)



# ## Data Processing

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

price_scl = MinMaxScaler()
price_data = data.iloc[:, 7:67].values
price_data = price_scl.fit_transform(price_data)
x_price = lookback_price(price_data, lookback)


label_scl = MinMaxScaler()
label_data = data.iloc[:, 67:82].values
label_data = label_scl.fit_transform(label_data)
y = lookback_label(label_data, 7)



n_samples = y.shape[0]
p = int(n_samples * 0.8)
q = p + int(n_samples * 0.1)


x_news_train = []
x_news_valid = []
x_news_test = []
x_news_3_valid=[]
for i in range(lookback):
    x_news_train.append(x_news[i][:p])
    x_news_valid.append(x_news[i][p:q])
    x_news_3_valid.append(x_news[i][p:p+3])
    x_news_test.append(x_news[i][q:])


x_price_train = x_price[:p]
x_price_valid = x_price[p:q]
x_price_3_valid=x_price[p:p+3]
x_price_test = x_price[q:]

y_train = y[:p]
y_valid = y[p:q]
y_3_valid= y[p:p+3]
y_test = y[q:]

#ToDo: Unzip code  shutil.unpack_archive("Data/Train/Under_10_min_training/data.zip",extract_dir="temp")
DATA_DIR = "Data/Train/Under_10_min_training/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
np.save(TEMP_DATA_DIR + "/x_news_train.npy", x_news_train)
np.save(TEMP_DATA_DIR + "/x_price_train.npy",x_price_train)
np.save(TEMP_DATA_DIR + "/y_train.npy", y_train)
shutil.make_archive("data","zip",TEMP_DATA_DIR)

if not os.path.exists(DATA_DIR+"data.zip"):
    shutil.move("data.zip",DATA_DIR)
else:
    shutil.copy("data.zip", DATA_DIR)
    os.remove("data.zip")
for file in os.listdir(TEMP_DATA_DIR):
    os.remove(TEMP_DATA_DIR+"/"+file)



#ToDo: Unzip code  shutil.unpack_archive("Data/Train/Under_90_min_tuning/data.zip",extract_dir="temp")
DATA_DIR = "Data/Train/Under_90_min_tuning/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
np.save(TEMP_DATA_DIR + "/x_news_train.npy", x_news_train)
np.save(TEMP_DATA_DIR + "/x_price_train.npy",x_price_train)
np.save(TEMP_DATA_DIR + "/y_train.npy", y_train)
shutil.make_archive("data","zip",TEMP_DATA_DIR)

if not os.path.exists(DATA_DIR+"data.zip"):
    shutil.move("data.zip",DATA_DIR)
else:
    shutil.copy("data.zip", DATA_DIR)
    os.remove("data.zip")
for file in os.listdir(TEMP_DATA_DIR):
    os.remove(TEMP_DATA_DIR+"/"+file)



#ToDo: Unzip code  shutil.unpack_archive("Data/Train/Best_hyperparameter_80_percent/data.zip",extract_dir="temp")
DATA_DIR = "Data/Train/Best_hyperparameter_80_percent/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
np.save(TEMP_DATA_DIR + "/x_news_train.npy", x_news_train)
np.save(TEMP_DATA_DIR + "/x_price_train.npy",x_price_train)
np.save(TEMP_DATA_DIR + "/y_train.npy", y_train)
shutil.make_archive("data","zip",TEMP_DATA_DIR)

if not os.path.exists(DATA_DIR+"data.zip"):
    shutil.move("data.zip",DATA_DIR)
else:
    shutil.copy("data.zip", DATA_DIR)
    os.remove("data.zip")
for file in os.listdir(TEMP_DATA_DIR):
    os.remove(TEMP_DATA_DIR+"/"+file)



#ToDo:Unzip code  shutil.unpack_archive("Data/Validation/3_samples/data.zip",extract_dir="temp")
#**********************************************************************************************
#TODO:Needs check
DATA_DIR = "Data/Validation/3_samples/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
np.save(TEMP_DATA_DIR + "/x_news_3_valid.npy", x_news_3_valid)
np.save(TEMP_DATA_DIR + "/x_price_3_valid.npy", x_price_3_valid)
np.save(TEMP_DATA_DIR + "/y_3_valid.npy", y_3_valid)
shutil.make_archive("data", "zip", TEMP_DATA_DIR)
if not os.path.exists(DATA_DIR + "data.zip"):
    shutil.move("data.zip", DATA_DIR)
else:
    shutil.copy("data.zip", DATA_DIR)
    os.remove("data.zip")
for file in os.listdir(TEMP_DATA_DIR):
    os.remove(TEMP_DATA_DIR +"/"+ file)


#**********************************************************************************




#ToDO: Unzip code  shutil.unpack_archive("Data/Validation/Validation_10_percent/data.zip",extract_dir="temp")
DATA_DIR = "Data/Validation/Validation_10_percent/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
np.save(TEMP_DATA_DIR + "/x_news_valid.npy", x_news_valid)
np.save(TEMP_DATA_DIR + "/x_price_valid.npy", x_price_valid)
np.save(TEMP_DATA_DIR + "/y_valid.npy", y_valid)
shutil.make_archive("data", "zip", TEMP_DATA_DIR)
if not os.path.exists(DATA_DIR + "data.zip"):
    shutil.move("data.zip", DATA_DIR)
else:
    shutil.copy("data.zip", DATA_DIR)
    os.remove("data.zip")
for file in os.listdir(TEMP_DATA_DIR):
    os.remove(TEMP_DATA_DIR +"/"+ file)




#ToDO:Unzip code  shutil.unpack_archive("Data/Test/Test_10_percent/data.zip",extract_dir="temp")
DATA_DIR = "Data/Test/Test_10_percent/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
np.save(TEMP_DATA_DIR + "/x_news_test.npy", x_news_test)
np.save(TEMP_DATA_DIR + "/x_price_test.npy", x_price_test)
np.save(TEMP_DATA_DIR + "/y_test.npy", y_test)
shutil.make_archive("data", "zip",TEMP_DATA_DIR)
if not os.path.exists(DATA_DIR + "data.zip"):
    shutil.move("data.zip", DATA_DIR)
else:
    shutil.copy("data.zip", DATA_DIR)
    os.remove("data.zip")
for file in os.listdir(TEMP_DATA_DIR):
    os.remove(TEMP_DATA_DIR +"/"+ file)



if not os.path.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)
joblib.dump(price_scl, TEMP_DATA_DIR + '/price_scl.scl')
joblib.dump(label_scl, TEMP_DATA_DIR + '/label_scl.scl')



os.remove('Data/All_new.csv')

