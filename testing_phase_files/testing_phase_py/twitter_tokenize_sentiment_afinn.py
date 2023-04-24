#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:56:04 2023

@author: oliviarenfro
"""

# %% Sentiment Scores 


# %% Package Loading

import pandas as pd
import copy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from textblob import TextBlob
from spellchecker import SpellChecker
#nltk.download('punkt') #string splitting
#nltk.download('stopwords') #stop words
#pip install afinn
from afinn import Afinn
afinn = Afinn(language='en') #set language to english 


# %% PREPROCESSING FOR SENTIMENT ANALYSIS

# %% DATA COPY

# deepcopy of imported data 
df3 = pd.read_csv("/Users/oliviarenfro/Desktop/thesis/data/original_data/base_df")
df4 = copy.deepcopy(df3)

# %% LOWER CASE

# review case to lower 
df4.reviewText = df4.reviewText.apply(str)
df4['reviewText'] = df4.apply(lambda row: (row['reviewText'].lower()), axis=1)  # lower case

# %% TOKENIZED 

# creates tokenized reviews
tknzr = TweetTokenizer()
df4['tokenized_review'] = df4.apply(lambda row: tknzr.tokenize(row['reviewText']), axis=1)  #tokenized review

# %% REMOVE STOP WORDS

# removing stop words and punctuation 
stop_words = (set(stopwords.words('english')))
punctuation = [",", ".", "!", "...", "?", " ", '\'', '\"', '\\', '(',')']
for i in punctuation:
    stop_words.add(i)
df4['tokenized_review'] = df4['tokenized_review'].apply(set)
df4['tokenized_review_no_stop'] = df4['tokenized_review'] - stop_words

# %% CORRECTED SPELLING; ERROR COUNT; SIG WORD COUNT

# counting and correcting spelling errors; takes a minute to run 
num_mispelled_words = []
corrected_words = []
num_words =[]
spell = SpellChecker()
for i in range(len(df4)):
    x = len(list(spell.unknown(df4.tokenized_review_no_stop[i])))
    y = len(df4.tokenized_review_no_stop[i])
    num_mispelled_words.append(x)
    num_words.append(len(df4.tokenized_review_no_stop[i]))
    words = []
    for j in df4.tokenized_review_no_stop[i]:
        textBlb = TextBlob(j)           
        words.append(str(textBlb.correct()))
    corrected_words.append([words])

df4['num_mispelled_words'] = num_mispelled_words
df4['corrected_spelling'] = corrected_words
df4["num_significant_words"] = num_words
for i in range(len(df4)): df4['corrected_spelling'][i] = df4.corrected_spelling[i][0]

# %% AFINN SENTIMENT

def afinn_average(lst):
    sum = 0
    #lst = nltk.word_tokenize(comment)
    for i in lst:
        print(i)
        sum = sum + afinn.score(i) 
    return sum/len(lst)

avg_sentiment_value = []
for i in range(len(df4)):
    sum = 0
    for j in df4.corrected_spelling[i]:
        sum = sum + afinn.score(j) 
    avg = sum/len(df4.corrected_spelling[i])
    avg_sentiment_value.append(avg)

sum_sentiment_value = []
for i in range(len(df4)):
    sum = 0
    for j in df4.corrected_spelling[i]:
        sum = sum + afinn.score(j) 
    sum_sentiment_value.append(sum)   
    
df4['avg_sentiment_score'] = avg_sentiment_value
df4['sum_sentiment_score'] = sum_sentiment_value

df4['vote'] = df4['vote'].fillna(0)

# %% SAVING SENTIMENT DF

sentiment_df = df4[['overall', 'verified', 'reviewTime','reviewerID','asin','reviewText','corrected_spelling',
                    'vote','num_mispelled_words','num_significant_words','avg_sentiment_score','sum_sentiment_score',
                    'summary','unixReviewTime','image', 'tokenized_review']].copy()


sentiment_df.to_csv("/Users/oliviarenfro/Desktop/thesis/data/processed_data/sentiment_df", index = False)

#import matplotlib.pyplot as plt
#plt.scatter(sentiment_df.overall, sentiment_df.avg_sentiment_score)
#plt.show()

text = list(sentiment_df.corrected_spelling)
whales = [nltk.FreqDist(i) for i  in text]
