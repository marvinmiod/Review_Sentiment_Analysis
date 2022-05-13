<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:21:46 2022
deployment of model

@author: user
"""


from tensorflow.keras.models import load_model
import os
import re
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sentiment_analysis_module import ExploratoryDataAnalysis, ModelCreation
from tensorflow.keras.preprocessing.text import tokenizer_from_json

#MODEL_PATH = os.path.join(os.getcwd(),  )

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_SentimentAnalysis.h5')

sentiment_classifier = load_model(MODEL_SAVE_PATH)
sentiment_classifier.summary()



#%% Tokenizer loading
# must import json first

JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)

#%% Loading of Data

# must be in list
# change review to list
#new_review = ['I think the first one hour is interesting, but the second half\
#    of the movie is boring, this movie just wasted two precious hours of my time\
#        and hard earned money. This  movie should be banned to avoid time being \
#            wasted']
            
# to get input from user you can use
new_review = [input('Provide your review about the movie\n\n')]
            
#%% EDA

# Step 1) Clean the data

eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

# to vectorize the new review
# to feed the tokens into keras
loaded_tokenizer = tokenizer_from_json(token)

# to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequences(new_review)

#%% Model Prediction

outcome = sentiment_classifier.predict(np.expand_dims(new_review, axis=-1)) # size (1,200)
print(np.argmax(outcome))

# sentiment value
# positive = [0,1]
# negative = [1,0]
#to replace the value 1 or 0 to positive or negative
sentiment_dict = {1: 'positive', 0: 'negative'}
print('this review is ' + sentiment_dict[np.argmax(outcome)])

=======
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:21:46 2022
deployment of model

@author: user
"""


from tensorflow.keras.models import load_model
import os
import re
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sentiment_analysis_module import ExploratoryDataAnalysis, ModelCreation
from tensorflow.keras.preprocessing.text import tokenizer_from_json

#MODEL_PATH = os.path.join(os.getcwd(),  )

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_SentimentAnalysis.h5')

sentiment_classifier = load_model(MODEL_SAVE_PATH)
sentiment_classifier.summary()



#%% Tokenizer loading
# must import json first

JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)

#%% Loading of Data

# must be in list
# change review to list
#new_review = ['I think the first one hour is interesting, but the second half\
#    of the movie is boring, this movie just wasted two precious hours of my time\
#        and hard earned money. This  movie should be banned to avoid time being \
#            wasted']
            
# to get input from user you can use
new_review = [input('Provide your review about the movie\n\n')]
            
#%% EDA

# Step 1) Clean the data

eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

# to vectorize the new review
# to feed the tokens into keras
loaded_tokenizer = tokenizer_from_json(token)

# to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequences(new_review)

#%% Model Prediction

outcome = sentiment_classifier.predict(np.expand_dims(new_review, axis=-1)) # size (1,200)
print(np.argmax(outcome))

# sentiment value
# positive = [0,1]
# negative = [1,0]
#to replace the value 1 or 0 to positive or negative
sentiment_dict = {1: 'positive', 0: 'negative'}
print('this review is ' + sentiment_dict[np.argmax(outcome)])

>>>>>>> 682604ece27cccceb5a2603476302552306d2e6e
