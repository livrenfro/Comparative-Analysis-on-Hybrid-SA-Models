#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:53:16 2023

@author: oliviarenfro
"""

# %% Package Loading

import pandas as pd
import copy

# %% DATA COPY

# deepcopy of imported data 
df5 = pd.read_csv("/Users/oliviarenfro/Desktop/thesis/data/processed_data/sentiment_df")
df6 = copy.deepcopy(df5)

# %% OUTLINE/TO-DO LIST

# DF: group by user, count total number of product reviews 
# DF: group by user, count number of related product reviews (categories in METADATA), ASIN is key
# verified? (group by userID, productID(asin), )
# image? (T if present, F if NULL)
# vote (range givves diff values)
# ratio of mispelled words (num_mispelled words/len(tokenized_review))
    # take tokenized_review ans remove punctuation
    # punctuation = [",", ".", "!", "...", "?", " ", '\'', '\"', '\\', '(',')']
    # df6['tokenized_review_no_stop'] = df6['tokenized_review'] - punctuation


# %% DF: NUMBER OF PRODUCT REVIEWS PER USER

num_reviews_per_user = df6.groupby('reviewerID')['asin'].count()
num_reviews_per_user = num_reviews_per_user.rename(index={0: "reviewerID", 1: "num_reviews_per_user"})

# can join using reveiwerID

# %% DF: NUMBER OF RELATED PRODUCT REVIEWS PER USER

# join metadata, join on asin and add categories 
# count number of product reviews where product has same category
# might have to expand/delist categories
    # for each reviewerID, count number of reviews that have nonzero intersection of category lists
    
# can join using reveiwerID
# %%  OTHER VARIABLE TRANSFORMATIONS


    
