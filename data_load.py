#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 01:33:58 2023

@author: oliviarenfro
"""

# PACKAGE IMPORT
import pandas as pd
import regex as re
import contractions
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#from nltk.stem import PorterStemmer #lemmatization better than stemming
#from nltk.stem import WordNetLemmatizer #lemma in next file
#from textblob import TextBlob #spellchecker was more accurate 
from spellchecker import SpellChecker
from nltk.corpus import wordnet
from string import punctuation
import numpy as np
import re
 
#nltk.download('punkt') #string splitting
#nltk.download('stopwords') #stop words


# DATA READ; two files, cats and dogs, scraped from Twitter
cats_df = pd.read_csv('./data/original_data/dataset_twitter-scraper_2023-cats.csv') #35 hashtags
cats_length = len(cats_df)

dogs_df = pd.read_csv('./data/original_data/dataset_twitter-scraper_2023-dogs.csv') #26 hashtags
dogs_length = len(dogs_df)
 


#%% HASHTAGS

#Creating new hastag column, dropping expanded columns; merging imported dfs

# CATS 
string = 'cats_df[\'hashtags\'] = cats_df[\'hashtags/0\'].fillna(\'na\')'
for i in range(1,36):
    string += ' + \',\' + cats_df[\'hashtags/%i\'].fillna(\'na\')' % i 
exec(string)

# splitting string into comma seperated list 
for i in range(0,cats_length):
    cats_df['hashtags'].iloc[i] = cats_df['hashtags'].iloc[i].split(",")

#filtering out na from hastag list
for i in range(0,cats_length):
    cats_df['hashtags'].iloc[i] = [x for x in cats_df['hashtags'].iloc[i] if x != 'na']
    
# Dropping hashtag columns 
string2 = 'cats_df = cats_df.drop(columns = [\'hashtags/0\' '
for i in range(1,36):
    string2 += ', \'hashtags/%i\'' % i
string2 += '])'
exec(string2)


# DOGS 
string = 'dogs_df[\'hashtags\'] = dogs_df[\'hashtags/0\'].fillna(\'na\')'
for i in range(1,27):
    string += ' + \',\' + dogs_df[\'hashtags/%i\'].fillna(\'na\')' % i 
exec(string)

# splitting string into comma seperated list 
for i in range(0,dogs_length):
    dogs_df['hashtags'].iloc[i] = dogs_df['hashtags'].iloc[i].split(",")

#filtering out na from hastag list
for i in range(0,dogs_length):
    dogs_df['hashtags'].iloc[i] = [x for x in dogs_df['hashtags'].iloc[i] if x != 'na']
    
# Dropping hashtag columns 
string2 = 'dogs_df = dogs_df.drop(columns = [\'hashtags/0\' '
for i in range(1,27):
    string2 += ', \'hashtags/%i\'' % i
string2 += '])'
exec(string2)

# joining dfs now with same column 
x = pd.concat([dogs_df,cats_df])
y = x.loc[x.astype(str).drop_duplicates().index]
df = y.reset_index(drop = True)
length = len(df)
df.to_pickle("./pickles/df_original.pkl")

#%% FULL_TEXT

# LOWERCASE, REMOVE EMOJIS 
df['full_text'] = df.apply(lambda row: (clean(row['full_text'],no_emoji = True, lower = True)), axis=1) 

# EXPANDING CONTRACTIONS 
text = df['full_text']
expanded_text = []
for i in range(0,length):
    split = text[i].split()
    split_length = len(split)
    entry_i = [] 
    for j in range(split_length):
        entry_i.append((contractions.fix(split[j])))
    entry_i = ' '.join(((entry_i[n]) for n in range(0, split_length)))
    expanded_text.append(entry_i)
   

# REMOVING URLS AND MENTIONS AND HASHTAGS
for i in range(length):
    expanded_text[i] = re.sub(r'http\S+', '', expanded_text[i])
    expanded_text[i] = re.sub(r'#\S+', '', expanded_text[i])
    expanded_text[i] = re.sub(r'@\S+', '', expanded_text[i])
    expanded_text[i] = re.sub(r'[0-9]', "", expanded_text[i])
    expanded_text[i] = re.sub(r"^\s+|\s+$", "", expanded_text[i])
df['full_text'] = expanded_text
 
# TOKENIZING
tknzr = TweetTokenizer()
df['tokenized_text'] = df.apply(lambda row: tknzr.tokenize(row['full_text']), axis=1) 

# Negation handling 
def Negation(sentence):	
  '''
  Input: Tokenized sentence (List of words)
  Output: Tokenized sentence with negation handled (List of words)
  '''
  temp = int(0)
  for i in range(len(sentence)):
      if sentence[i-1] in ['not',"n't", 'no']:
          antonyms = []
          for syn in wordnet.synsets(sentence[i]):
              syns = wordnet.synsets(sentence[i])
              w1 = syns[0].name()
              temp = 0
              for l in syn.lemmas():
                  if l.antonyms():
                      antonyms.append(l.antonyms()[0].name())
              max_dissimilarity = 0
              for ant in antonyms:
                  syns = wordnet.synsets(ant)
                  w2 = syns[0].name()
                  syns = wordnet.synsets(sentence[i])
                  w1 = syns[0].name()
                  word1 = wordnet.synset(w1)
                  word2 = wordnet.synset(w2)
                  if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                      temp = 1 - word1.wup_similarity(word2)
                  if temp>max_dissimilarity:
                      max_dissimilarity = temp
                      antonym_max = ant
                      sentence[i] = antonym_max
                      sentence[i-1] = ''
  while '' in sentence:
      sentence.remove('')
  return sentence

negated = []
for i in range(length): 
    negated.append(Negation(df['tokenized_text'][i]))
df['tokenized_text'] = negated 

# REMOVING STOP WORDS AND PUNCTUATION 
stop_words = set(stopwords.words('english')) - {'not', 'against', 'nor', 'won', 'no'} 
stop_words.add('...')
stop_words.add("'s")
punctuation = set(punctuation)
for i in punctuation:
    stop_words.add(i)
tokenized_text_no_stop = []
for i in range(0,length):
    tokenized_text_no_stop.append([x for x in df['tokenized_text'][i] if x not in stop_words])
df['tokenized_text_no_stop'] = tokenized_text_no_stop

#ADDING HASHTAG TEXT IN
for i in range(length):
    for j in range(len(df.hashtags[i])):
        df['tokenized_text_no_stop'][i].append(df.hashtags[i][j].lower())
        
#%% CORRECTING SPELLING 

# counting and correcting spelling errors; takes a minute to run 
num_mispelled_words = []
corrected_words = []
num_words =[]
spell = SpellChecker()
for i in range(0,length):
    x = len(list(spell.unknown(df.tokenized_text_no_stop[i])))
    y = len(df.tokenized_text_no_stop[i])
    num_mispelled_words.append(x)
    num_words.append(len(df.tokenized_text_no_stop[i]))
    words = []
    for j in range(len(df.tokenized_text_no_stop[i])):        
        if spell.correction(df.tokenized_text_no_stop[i][j]) == None:
            words.append(df.tokenized_text_no_stop[i][j])
        else:
            words.append((spell.correction(df.tokenized_text_no_stop[i][j])))
    #  print(words)
    corrected_words.append([words])
    
df['num_mispelled_words'] = num_mispelled_words
df['corrected_spelling'] = corrected_words
df["num_significant_words"] = num_words

for i in range(length): df['corrected_spelling'][i] = df.corrected_spelling.iloc[i][0]


#%% saving variables for environment 
df_cleaned = df[['id', 'full_text', 'hashtags','tokenized_text','tokenized_text_no_stop','num_mispelled_words','corrected_spelling','num_significant_words']]

df.to_pickle("./pickles/df.pkl")
df_cleaned.to_pickle('./pickles/df_cleaned.pkl')