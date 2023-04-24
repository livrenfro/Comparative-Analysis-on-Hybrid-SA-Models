#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 9 12:49:34 2023

@author: oliviarenfro
"""

from nltk.parse import CoreNLPParser
from nltk.stem.wordnet import WordNetLemmatizer
#import ast
import copy
import pandas as pd
from nltk.corpus import wordnet

processed_df = pd.read_pickle('./pickles/df_cleaned.pkl')

#processed_df = processed_df.drop([628])
processed_df = processed_df[processed_df['corrected_spelling'].map(lambda d: len(d)) > 0]
processed_df.reset_index(drop=True, inplace=True)
length =len(processed_df)

# %%

# connect to server in temrinal after stanford-corenlp-4.5.4 download from: https://stanfordnlp.github.io/CoreNLP/download.html
# once dowmloaded, run the following commands in the temrinal to establish a connction
# cd stanford-corenlp-4.5.4
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000'

# after a successful connection has been made, return to this file and continue runnng 

# initaliazing tagger
parser = CoreNLPParser(url='http://localhost:9000')
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

# creating POS tags 
tagged = copy.copy(processed_df['corrected_spelling'])
for i in range(len(tagged)):
    tagged[i] = pos_tagger.tag(tagged[i])


# changing into wordnet compatible
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    
# lemmatizing with tag
lemmatizer = WordNetLemmatizer()
lemma_tag = []
tag_wordnet = []
for i in range(len(tagged)):
    vector = []
    wntag_vec = []
    for j in range(len(tagged[i])):
        tag = tagged[i][j][1]
        #print(tag)
        word = tagged[i][j][0]
        #print(word)
        wntag = get_wordnet_pos(tag)
        wntag_vec.append(wntag)
        if wntag == '': # not supply tag in case of None
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, wntag) 
        vector.append(lemma)
        #print('vector is = ', vector)
    lemma_tag.append(vector)
    tag_wordnet.append(wntag_vec)
    #print('lemma_tag is = ', lemma_tag)
processed_df['tag_wordnet']  = tag_wordnet    
processed_df['lemmatized']  = lemma_tag

#%% saving variables for enviroment
processed_df.to_pickle("./pickles/processed_df.pkl")