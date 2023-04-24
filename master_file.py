#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:47:07 2023

@author: oliviarenfro
"""

# setting working directory
import os
wd=os.getcwd()
os.chdir(wd)


#%%  STAGE 1: data cleaning and preprocessing STAGE 1

# fixing hashtags, urls, mentions, special characters, emojis, capitalization, stop words, 
# spelling, expanding contractions, tokenizing, counting mispelled words for later analysis 

with open("/Users/oliviarenfro/Desktop/thesis/data_load.py") as t:
    exec(t.read())
    
# to do: create unique id column; let other files build supplementary dataframes instead of working on same data

#%%  STAGE 2: data tagging and lemmatizing

# POS tagging using stanford java API (literature review: most accurate), tag transformation for 
# compatibility with WordNet lemmatizer, and lemmatization


with open("/Users/oliviarenfro/Desktop/thesis/tagging_script.py") as f:
    exec(f.read())
    
    
#%%  STAGE 3: sentiwordnet sentiment

# sentiment = sum positively scored words in text minus sum of negatively scored ones (average skewed by zero senitment)

with open("/Users/oliviarenfro/Desktop/thesis/sentiword_sentiment.py") as w:
    exec(w.read())