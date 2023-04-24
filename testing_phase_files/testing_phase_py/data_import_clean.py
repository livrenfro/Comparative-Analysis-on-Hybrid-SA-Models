#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:31:33 2023

@author: oliviarenfro
"""

# %% Data Import/Cleaning 

# %% BASE DF
import pandas as pd

df = pd.read_json('/Users/oliviarenfro/Desktop/thesis/data/original_data/AMAZON_FASHION_5.json',  lines=True)
df2 = df.drop(["style"], axis='columns') #drops style column, redundant, not good 
df3 = df2.drop_duplicates(subset=['reviewerID', 'unixReviewTime'], keep='last') # drops duplicates of reviewerID & unixReviewTime
df3 = df3.drop_duplicates(subset=['reviewText'], keep='last')
df3.to_csv("/Users/oliviarenfro/Desktop/thesis/data/original_data/base_df", index=False)


# %% META DATA

import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
# !wget http://deepyeti.ucsd.edu/jianmo/amazon/sample/meta_Computers.json.gz RUN IN TERMINAL

### load the meta data

data = []
with gzip.open('meta_Computers.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
    #CHANGE NAMES OF DF
# convert list into pandas dataframe
df = pd.DataFrame.from_dict(data)

### remove rows with unformatted title (i.e. some 'title' may still contain html style content)
df3 = df.fillna('')
df4 = df3[df3.title.str.contains('getTime')] # unformatted rows
df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows


