# -*- coding: utf-8 -*-

import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import pickle
import numpy as np
#nltk.download('words')
words = set(nltk.corpus.words.words())
#importing enviroment variables 
processed_df = pickle.load(open("./pickles/processed_df.pkl", "rb"))


#%% VADER scoring and seperating dictionary results

VADER_scores = pd.DataFrame()   
VADER_scores['id'] = processed_df['id']
VADER_scores['clean_tweets'] = processed_df['tokenized_text'].map(lambda row: ' '.join(row))

sid = SentimentIntensityAnalyzer()
VADER_scores_vec = VADER_scores.apply(lambda row: sid.polarity_scores(row['clean_tweets']), axis=1)  
VADER_scores['VADER_total_sentiment'] = VADER_scores_vec.apply(lambda score_dict: score_dict['compound'])
VADER_scores['VADER_pos_sentiment'] = VADER_scores_vec.apply(lambda score_dict: score_dict['pos'])
VADER_scores['VADER_neg_sentiment'] = VADER_scores_vec.apply(lambda score_dict: score_dict['neg'])
VADER_scores['VADER_neu_sentiment'] = VADER_scores_vec.apply(lambda score_dict: score_dict['neu'])
#VADER_scores = VADER_scores.drop('VADER_scores', axis = 1)


VADER_scores['VADER_tagged_sentiment'] = np.where(VADER_scores['VADER_total_sentiment'] > 0.5, 'POS', np.where(VADER_scores['VADER_total_sentiment'] < -0.5, 'NEG', 'NEU'))

#%% saving variables for environment

VADER_scores.to_pickle("./pickles/VADER_scores.pkl")