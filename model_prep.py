#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 09:29:43 2023

@author: oliviarenfro
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim.downloader
from sklearn.preprocessing import scale
processed_df = pickle.load(open("./pickles/processed_df.pkl", "rb"))
VADER_scores = pickle.load(open("./pickles/VADER_scores.pkl", "rb"))
sentinet_scores = pickle.load(open("./pickles/sentinet_scores.pkl", "rb"))


#%% SPLITTING DATA 

df = pd.DataFrame([processed_df['id'], processed_df['lemmatized'], sentinet_scores['sentinet_tagged_sentiment']]).transpose()
df['vader_tagged_sentiment'] = VADER_scores['VADER_tagged_sentiment']

# vader 
df_pos_vader = df[df['vader_tagged_sentiment'] == 'POS']
df_neg_vader = df[df['vader_tagged_sentiment'] == 'NEG']
df_neu_vader = df[df['vader_tagged_sentiment'] == 'NEU']

# vader split
#pos
X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(df_pos_vader['lemmatized'], 
                                                                      df_pos_vader['vader_tagged_sentiment'], 
                                                                      test_size=0.25, random_state=30)
#neg
X_train_neg, X_test_neg, Y_train_neg, Y_test_neg = train_test_split(df_neg_vader['lemmatized'], 
                                                                      df_neg_vader['vader_tagged_sentiment'], 
                                                                      test_size=0.25, random_state=30)
#neu
X_train_neu, X_test_neu, Y_train_neu, Y_test_neu = train_test_split(df_neu_vader['lemmatized'], 
                                                                      df_neu_vader['vader_tagged_sentiment'], 
                                                                      test_size=0.25, random_state=30)
#together 
X_train_vader = (X_train_pos.append(X_train_neg)).append(X_train_neu)
X_train_vader = X_train_vader.reset_index().drop(['index'], axis = 1)

X_test_vader = (X_test_pos.append(X_test_neg)).append(X_test_neu)
X_test_vader = X_test_vader.reset_index().drop(['index'], axis = 1)

Y_train_vader = (Y_train_pos.append(Y_train_neg)).append(Y_train_neu)
Y_train_vader = Y_train_vader.reset_index().drop(['index'], axis = 1)

Y_test_vader = (Y_test_pos.append(Y_test_neg)).append(Y_test_neu)
Y_test_vader = Y_test_vader.reset_index().drop(['index'], axis = 1)

# sentinet 
df_pos_sentinet = df[df['sentinet_tagged_sentiment'] == 'POS']
df_neg_sentinet = df[df['sentinet_tagged_sentiment'] == 'NEG']
df_neu_sentinet = df[df['sentinet_tagged_sentiment'] == 'NEU']

# sentinet split
#pos
X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(df_pos_sentinet['lemmatized'], 
                                                                      df_pos_sentinet['sentinet_tagged_sentiment'], 
                                                                      test_size=0.25, random_state=30)
#neg
X_train_neg, X_test_neg, Y_train_neg, Y_test_neg = train_test_split(df_neg_sentinet['lemmatized'], 
                                                                      df_neg_sentinet['sentinet_tagged_sentiment'], 
                                                                      test_size=0.25, random_state=30)
#neu
X_train_neu, X_test_neu, Y_train_neu, Y_test_neu = train_test_split(df_neu_sentinet['lemmatized'], 
                                                                      df_neu_sentinet['sentinet_tagged_sentiment'], 
                                                                      test_size=0.25, random_state=30)
#together 
X_train_sentinet = (X_train_pos.append(X_train_neg)).append(X_train_neu)
X_train_sentinet = X_train_sentinet.reset_index().drop(['index'], axis = 1)

X_test_sentinet = (X_test_pos.append(X_test_neg)).append(X_test_neu)
X_test_sentinet = X_test_sentinet.reset_index().drop(['index'], axis = 1)

Y_train_sentinet = (Y_train_pos.append(Y_train_neg)).append(Y_train_neu)
Y_train_sentinet = Y_train_sentinet.reset_index().drop(['index'], axis = 1)

Y_test_sentinet = (Y_test_pos.append(Y_test_neg)).append(Y_test_neu)
Y_test_sentinet = Y_test_sentinet.reset_index().drop(['index'], axis = 1)


with open('./pickles/Y_train_sentinet.pkl','wb') as f:
     pickle.dump(Y_train_sentinet, f)
with open('./pickles/Y_test_sentinet.pkl','wb') as f:
     pickle.dump(Y_test_sentinet, f)  
with open('./pickles/Y_train_vader.pkl','wb') as f:
     pickle.dump(Y_train_vader, f)
with open('./pickles/Y_test_vader.pkl','wb') as f:
     pickle.dump(Y_test_vader, f)

#%% TFIDF TRANSFORMATION

# TFIDF REQUIRES STRING:
vectorizer= TfidfVectorizer()

#tf_X_train_vader
pre_processed = []
pre_process = X_train_vader.lemmatized
for i in range(0,len(pre_process)):
    pre_processed.append(' '.join(v for v in pre_process[i]))
tf_X_train_vader = vectorizer.fit_transform(pre_processed)
with open('./pickles/tf_X_train_vader.pkl','wb') as f:
     pickle.dump(tf_X_train_vader, f)

#tf_X_test_vader
pre_processed = []
pre_process = X_test_vader.lemmatized
for i in range(0,len(pre_process)):
    pre_processed.append(' '.join(v for v in pre_process[i]))
tf_X_test_vader = vectorizer.transform(pre_processed)
with open('./pickles/tf_X_test_vader.pkl','wb') as f:
     pickle.dump(tf_X_test_vader, f)  

#tf_X_train_sentinet
pre_processed = []
pre_process = X_train_sentinet.lemmatized
for i in range(0,len(pre_process)):
    pre_processed.append(' '.join(v for v in pre_process[i]))
tf_X_train_sentinet = vectorizer.fit_transform(pre_processed)
with open('./pickles/tf_X_train_sentinet.pkl','wb') as f:
     pickle.dump(tf_X_train_sentinet, f)
     
#X_test_sentinet
pre_processed = []
pre_process = X_test_sentinet.lemmatized
for i in range(0,len(pre_process)):
    pre_processed.append(' '.join(v for v in pre_process[i]))
tf_X_test_sentinet = vectorizer.transform(pre_processed)
with open('./pickles/tf_X_test_sentinet.pkl','wb') as f:
     pickle.dump(tf_X_test_sentinet, f)      

#%% WordEmbedding TRANSFORMATION

# PRETRAINED WORD EMBEDDING MODELS; DOWNLOAD AND SAVE 
glove_vectors = gensim.downloader.load('glove-twitter-200')
word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

#saving models after initial download
with open("./models/word2vec_glove200.model.pkl"  , 'wb') as file:  
    pickle.dump(glove_vectors, file) 
with open("./models/word2vec_google300.model.pkl"  , 'wb') as file:  
    pickle.dump(word2vec_vectors, file)   


# could not upload models to github, will have to keep these commented out and downlaod above
# loading models after initial download
#with open("./models/word2vec_glove200.model.pkl" , 'rb') as file:  
    #glove_vectors = pickle.load(file)
#with open("./models/word2vec_google300.model.pkl" , 'rb') as file:  
    #word2vec_vectors = pickle.load(file)

#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(model, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# using glove 200
glove200_X_train_vader = np.concatenate([buildWordVector(glove_vectors,z, 200) for z in X_train_vader.lemmatized])
glove200_X_train_vader = scale(glove200_X_train_vader)

glove200_X_test_vader = np.concatenate([buildWordVector(glove_vectors,z, 200) for z in X_test_vader.lemmatized])
glove200_X_test_vader = scale(glove200_X_test_vader)

glove200_X_train_sentinet = np.concatenate([buildWordVector(glove_vectors,z, 200) for z in X_train_sentinet.lemmatized])
glove200_X_train_sentinet = scale(glove200_X_train_sentinet)

glove200_X_test_sentinet = np.concatenate([buildWordVector(glove_vectors,z, 200) for z in X_test_sentinet.lemmatized])
glove200_X_test_sentinet = scale(glove200_X_test_sentinet)


#saving glove 200 data sets
with open('./pickles/glove200_X_train_vader.pkl','wb') as f:
     pickle.dump(glove200_X_train_vader, f)
with open('./pickles/glove200_X_test_vader.pkl','wb') as f:
     pickle.dump(glove200_X_test_vader, f)  
with open('./pickles/glove200_X_train_sentinet.pkl','wb') as f:
     pickle.dump(glove200_X_train_sentinet, f)
with open('./pickles/glove200_X_test_sentinet.pkl','wb') as f:
     pickle.dump(glove200_X_test_sentinet, f)

# using google 300
google300_X_train_vader = np.concatenate([buildWordVector(word2vec_vectors,z, 300) for z in X_train_vader.lemmatized])
google300_X_train_vader = scale(google300_X_train_vader)

google300_X_test_vader = np.concatenate([buildWordVector(word2vec_vectors,z, 300) for z in X_test_vader.lemmatized])
google300_X_test_vader = scale(google300_X_test_vader)

google300_X_train_sentinet = np.concatenate([buildWordVector(word2vec_vectors,z, 300) for z in X_train_sentinet.lemmatized])
google300_X_train_sentinet = scale(google300_X_train_sentinet)

google300_X_test_sentinet = np.concatenate([buildWordVector(word2vec_vectors,z, 300) for z in X_test_sentinet.lemmatized])
google300_X_test_sentinet = scale(google300_X_test_sentinet)

#saving google 300 data sets
with open('./pickles/google300_X_train_vader.pkl','wb') as f:
     pickle.dump(google300_X_train_vader, f)
with open('./pickles/google300_X_test_vader.pkl','wb') as f:
     pickle.dump(google300_X_test_vader, f)  
with open('./pickles/google300_X_train_sentinet.pkl','wb') as f:
     pickle.dump(google300_X_train_sentinet, f)
with open('./pickles/google300_X_test_sentinet.pkl','wb') as f:
     pickle.dump(google300_X_test_sentinet, f)