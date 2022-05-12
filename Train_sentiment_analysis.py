# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:44:23 2022

THis is a train.py file trains the model to determine if a review is positive
or negative


@author: user
"""

#%%

import re
import os
import pandas as pd
import numpy as np
from Sentiment_analysis_module import ExploratoryDataAnalysis, ModelCreation
from Sentiment_analysis_module import ModelEvaluation
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(),'Log_sentiment_analysis')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_SentimentAnalysis.h5')  

#%%

# EDA
df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

# Step 2) Data cleaning, remove tag 

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review) # to remove tags
review = eda.lower_split(review) # to convert to lower case and split

# step 3) Feature selection
# step 4) Data Vectorization
review = eda.sentiment_tokenizer(review, TOKENIZER_JSON_PATH)
review = eda.sentiment_pad_sequences(review)

# Step 5) Data Pre-processing
# One Hot Encoder
# One Hot Encoding for Label
one_hot_encoder = OneHotEncoder(sparse=False)
sentiment_encoded = one_hot_encoder.fit_transform(np.expand_dims(sentiment, 
                                                                 axis=-1))

# to calculate number of total categories
num_categories = len(np.unique(sentiment))

# x = review, y = sentiment
# train test split
x_train, x_test, y_train, y_test = train_test_split(review, 
                                                    sentiment_encoded,
                                                    test_size=0.3,
                                                    random_state=123)
# remember to convert to 3 Dimension for the input shape
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# to inverse the positive and negative review

# to know [0,1] is positive and [1,0] is negative
print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))

#%% Model Creation

mc = ModelCreation()
num_words = 10000

# choose either one file
model = mc.lstm_layer(num_words, num_categories)
#model = mc.simple_lstm_layer(num_words, num_categories)

log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# tensorboard callback
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

# early stopping callback
early_stopping_callback = EarlyStopping(monitor='loss', patience=2 )

#%% Compile & Model fitting

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='acc')


hist = model.fit(x_train, y_train, epochs=1, 
          validation_data=(x_test,y_test),
          callbacks=[tensorboard_callback, early_stopping_callback])

print(hist.history.keys())

#%% Model Evaluation

# preallocation of memory approach, use the below

#predicted_advanced = np.empty([len(x_test), num_categories])

# correct one is with output 2 (number categories)
predicted_advanced = np.empty([len(x_test), 2])


for index, test in enumerate(x_test):
    predicted_advanced[index:] = model.predict(np.expand_dims(test, axis=0))
    
#%% Model analysis    

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

model_eval = ModelEvaluation()
model_eval.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)
