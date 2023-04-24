#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 10:32:56 2023

@author: oliviarenfro
"""

#import nltk
#nltk.download('sentiwordnet')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import pandas as pd
import pickle

#importing environment variables 
processed_df = pickle.load(open("./pickles/processed_df.pkl", "rb"))
length = len(processed_df)


#%%
lemma = processed_df['lemmatized']
tag_wordnet = processed_df['tag_wordnet']
sentiment = []
pos_sentiment = []
neg_sentiment = []
for i in range(length):
    sentiment_pos = 0
    sentiment_neg = 0
    sentiment_obj = 0
    for j in range(len(lemma[i])):
        synsets = wn.synsets(lemma[i][j], pos=tag_wordnet[i][j])
        if not synsets:
            continue
        # take the first sense, the most common
        synset = synsets[0]
        #print(synset)
        swn_synset = swn.senti_synset(synset.name())
        #print(swn_synset)
        sentiment_pos += swn_synset.pos_score()
        sentiment_neg += swn_synset.neg_score()
    sentiment.append((sentiment_pos - sentiment_neg))
    pos_sentiment.append(sentiment_pos)
    neg_sentiment.append(sentiment_neg)



sentinet_scores = pd.DataFrame()    

sentinet_scores['id'] = processed_df['id']
sentinet_scores['sentinet_total_sentiment'] = sentiment
sentinet_scores['sentinet_pos_sentiment'] = pos_sentiment
sentinet_scores['sentinet_neg_sentiment'] = neg_sentiment
sentinet_scores['sentinet_tagged_sentiment'] = np.where(sentinet_scores['sentinet_total_sentiment'] > 0, 'POS', np.where(sentinet_scores['sentinet_total_sentiment'] < 0, 'NEG', 'NEU'))
#%% saving variables for environment

sentinet_scores.to_pickle("./pickles/sentinet_scores.pkl")