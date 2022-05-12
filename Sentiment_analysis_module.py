# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:53:00 2022

Training Model using classes

@author: user
"""


import re
import pandas as pd
import numpy as np
import re
import datetime
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential #model is only for Sequential Model
from tensorflow.keras.layers import Dropout, Dense,  LSTM
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras import Input
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

#%%

class ExploratoryDataAnalysis():
    def __init__(self): # reason why use __init__ to pass the data inside
        pass
    
    def remove_tags(self, data):
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>', ' ', text)
            
        return data
            
    def lower_split(self, data):
        """
        This function convert text into lower case and split into list
        Parameters
        ----------
        data : Array
            Raw training data contains strings.

        Returns
        -------
        data : List
            Cleaned data in list.

        """
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
     
        return data
    
    def sentiment_tokenizer(self, data, token_save_path,
                           num_words=10000,oov_token='<OOV>', prt=False):
        """
        This function to assign number to each word and save it in a token file
        as dictionaries

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        token_save_path : TYPE
            DESCRIPTION.
        num_words : TYPE, optional
            DESCRIPTION. The default is 10000.
        oov_token : TYPE, optional
            DESCRIPTION. The default is '<OOV>'.
        prt : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
         # OOV out of vocab
        
        # tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        # to save the tokenizer for deployment purposes
        
        token_json = tokenizer.to_json()
        
        with open(TOKENIZER_JSON_PATH, 'w') as json_file:
            json.dump(token_json, json_file)
        
        
        # to observe the number of words
        word_index = tokenizer.word_index
        
        if prt == True:
            print(word_index)
            print(dict(list(word_index.items())[0:10]))
        
        # to vectorize the sequence of text
            data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def sentiment_pad_sequences(self, data):
        return pad_sequences(data, maxlen=200,padding='post', 
                              truncating='post')
    
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self,num_words, num_categories,
                   embedding_output=64, nodes=32, dropout_value=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout_value)) # dropout layer
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout_value))
        # output correspend to the y_test column (15000, 2)
        # y has to predict the review and the sentiment
        model.add(Dense(num_categories, activation='softmax')) 
        model.summary()
        
        return model
        
    def simple_lstm_layer(self,num_words, num_categories,
                   embedding_output=64, nodes=32, dropout_value=0.2):
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout_value)) # dropout layer
        # output correspend to the y_test column (15000, 2)
        # y has to predict the review and the sentiment
        model.add(Dense(num_categories, activation='softmax')) 
        model.summary()
        
        return model
    
    # can add def RNN_layer also
                                                      
class ModelEvaluation():
    def report_metrics(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        # to print accuracy score its 56% and its bad, like tossing a coin
        print(accuracy_score(y_true, y_pred))
      
         
    
#%%


# path where the logfile for tensorboard call back
LOG_PATH = os.path.join(os.getcwd(),'Log_sentiment_analysis')
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
# Model save path
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_SentimentAnalysis.h5')  
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'


df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']
    
#%% EDA function call
# this is a test data

eda = ExploratoryDataAnalysis()
test = eda.remove_tags(review) # remove html tag save the review text in to test var
test = eda.lower_split(test) # 

test = eda.sentiment_tokenizer(test, token_save_path=TOKENIZER_JSON_PATH,
                              prt=True)

test = eda.sentiment_pad_sequences(test)

#%% Model creation function call

num_words=10000
# this one take the unique value in sentiment file, positive and negative,
# or num categories can add 2
num_categories = len(sentiment.unique()) 
mc = ModelCreation()
model = mc.lstm_layer(num_words, num_categories)
    
#%% Step 3: Data cleaning
# this code to find all the unwanted character in the text to be removed
# to remove html tags
