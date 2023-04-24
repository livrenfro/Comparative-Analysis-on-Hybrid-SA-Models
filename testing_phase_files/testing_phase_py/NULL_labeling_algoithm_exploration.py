#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:35:05 2023

@author: oliviarenfro
"""

# Practice text labelling package 

#%% set up 

# installed wget, fastText
#   brew install wget
#   wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
#   unzip v0.9.2.zip
#   cd fastText-0.9.2
#   pip install .

# imported data from stack exgance with tags (labelled)
#   wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz

# split data into training and validation 
#   head -n 12404 cooking.stackexchange.txt > cooking.train
#   tail -n 3000 cooking.stackexchange.txt > cooking.valid


#%% setting up model 

import fasttext
#help(fasttext.FastText)

#train model on trianing data
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train")
#save model 
model.save_model("model_cooking.bin")

#test model on sentences
#model.predict("Which baking dish is best to bake a banana bread ?")
#model.predict("Why not put knives in the dishwasher?")

#test model on validation set
model.test("/Users/oliviarenfro/cooking.valid")
# output: number of samples, precision, recall
model.test("/Users/oliviarenfro/cooking.valid", k=5)

#test model on top 5 tags 
#model.predict("Why not put knives in the dishwasher?", k=5)

# %% making model better 

#preprocess data 
#   cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
#   head -n 12404 cooking.preprocessed.txt > cooking.train
#   tail -n 3000 cooking.preprocessed.txt > cooking.valid

# PREPROCESSED DATA; training model on preprocessed data (no upper/lower case weirdness)
import fasttext
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train")
model.test("/Users/oliviarenfro/cooking.valid")

# EPOCH: raising epoch value increases number of times that the model sees the training data 
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train", epoch=25)
model.test("/Users/oliviarenfro/cooking.valid")

# LEANRING RATE: decreasing/increasing learning rate can also imporve model  (0.1-1.0 is good range)
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train", lr=1.0)
model.test("/Users/oliviarenfro/cooking.valid")

# EPOCH AND LEARNING RATE together:
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train", lr=1.0, epoch=25)
model.test("/Users/oliviarenfro/cooking.valid")

# WORD N-GRAMS
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train", lr=1.0, epoch=25, wordNgrams=2)
model.test("/Users/oliviarenfro/cooking.valid")

# SCALING UP FOR LARGE DATA; -loss hs paramenter;  hierarchical softmax instead of the regular softmax
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
model.test("/Users/oliviarenfro/cooking.valid")

# MULTILABEL CLASSIFICATION
# handle multiple labels using independent binary classifiers for each label; can be done with -loss one-vs-all or -loss ova
model = fasttext.train_supervised(input="/Users/oliviarenfro/cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
# decrease learning rate good for this loss function compared to others 
# TEST:predict as many labels as possible (k = -1) and only those w probability 50% or more (threshold = 0.5)
model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
model.test("/Users/oliviarenfro/cooking.valid", k=-1, threshold=0.5)

