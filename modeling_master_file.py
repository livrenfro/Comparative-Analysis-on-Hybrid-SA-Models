#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:49:50 2023

@author: oliviarenfro
"""


import numpy as np 
import numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD

   
#%% loading data  

# word embedding x data
#glove
with open('./pickles/glove200_X_train_vader.pkl','rb') as f:
    glove200_X_train_vader = pickle.load(f)
with open('./pickles/glove200_X_test_vader.pkl','rb') as f:
    glove200_X_test_vader = pickle.load(f)
with open('./pickles/glove200_X_train_sentinet.pkl','rb') as f:
    glove200_X_train_sentinet = pickle.load(f)
with open('./pickles/glove200_X_test_sentinet.pkl','rb') as f:
    glove200_X_test_sentinet = pickle.load(f)


#googleword2vec x data
with open('./pickles/google300_X_train_vader.pkl','rb') as f:
    google300_X_train_vader = pickle.load(f)
with open('./pickles/google300_X_test_vader.pkl','rb') as f:
    google300_X_test_vader = pickle.load(f)
with open('./pickles/google300_X_train_sentinet.pkl','rb') as f:
    google300_X_train_sentinet = pickle.load(f)
with open('./pickles/google300_X_test_sentinet.pkl','rb') as f:
    google300_X_test_sentinet = pickle.load(f)
    
# tfidf x data
with open('./pickles/tf_X_train_vader.pkl','rb') as f:
    tf_X_train_vader = pickle.load(f)
with open('./pickles/tf_X_test_vader.pkl','rb') as f:
    tf_X_test_vader = pickle.load(f)
with open('./pickles/tf_X_train_sentinet.pkl','rb') as f:
    tf_X_train_sentinet = pickle.load(f)
with open('./pickles/tf_X_test_sentinet.pkl','rb') as f:
    tf_X_test_sentinet = pickle.load(f)
    
# y data 
with open('./pickles/Y_train_sentinet.pkl','rb') as f:
    Y_train_sentinet = pickle.load(f)
with open('./pickles/Y_test_sentinet.pkl','rb') as f:
    Y_test_sentinet = pickle.load(f)
with open('./pickles/Y_train_vader.pkl','rb') as f:
    Y_train_vader = pickle.load(f)
with open('./pickles/Y_test_vader.pkl','rb') as f:
    Y_test_vader = pickle.load(f)


#%% PCA on large data (TF-IDF)

#  determingin best number of components 
def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components

# x  vader 
#finding best number of components with 0.95 var threshold
svd = TruncatedSVD(1700, random_state=42)
svd.fit_transform(tf_X_train_vader) 
tsvd_var_ratios = svd.explained_variance_ratio_
num_components_xvader = select_n_components(tsvd_var_ratios, 0.95)
#applying transformation 
svdv = TruncatedSVD(num_components_xvader, random_state=42)
svdv.fit_transform(tf_X_train_vader) 
tf_X_train_vader_pca = svdv.transform(tf_X_train_vader)
tf_X_test_vader_pca = svdv.transform(tf_X_test_vader)

# x sentinet
#finding best number fofcomponents with 0.95 var threshold
svd = TruncatedSVD(1700, random_state=40)
svd.fit_transform(tf_X_train_sentinet) 
tsvd_var_ratios = svd.explained_variance_ratio_
num_components_xsentinet = select_n_components(tsvd_var_ratios, 0.95)
#applying transformation 
svds = TruncatedSVD(num_components_xsentinet, random_state=40)
svds.fit_transform(tf_X_train_sentinet) 
tf_X_train_sentinet_pca = svds.transform(tf_X_train_sentinet)
tf_X_test_sentinet_pca = svds.transform(tf_X_test_sentinet)


#saving tf-idf svd sets
with open('./pickles/tf_X_train_vader_pca.pkl','wb') as f:
     pickle.dump(tf_X_train_vader_pca, f)
with open('./pickles/tf_X_test_vader_pca.pkl','wb') as f:
     pickle.dump(tf_X_test_vader_pca, f)
with open('./pickles/tf_X_train_sentinet_pca.pkl','wb') as f:
     pickle.dump(tf_X_train_sentinet_pca, f) 
with open('./pickles/tf_X_test_sentinet_pca.pkl','wb') as f:
     pickle.dump(tf_X_test_sentinet_pca, f)

#%% VALIDATION CHECK: parameter tuning 

#Random Forest; n_estimators
def RF_validation_curve(x_train, y_train, plotname):  
    parameter_range = [150, 300, 450, 600, 750, 900]
    train_scoreNum, test_scoreNum = validation_curve(
                                    RandomForestClassifier(),
                                    X = x_train, y = y_train, 
                                    param_name = 'n_estimators', 
                                    param_range = parameter_range, cv = 5)
    # Calculating mean of training score
    mean_train_score = np.mean(train_scoreNum, axis = 1)
    # Calculating mean of testing score
    mean_test_score = np.mean(test_scoreNum, axis = 1)
    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score,
         label = "Training Score", color = 'b')
    plt.plot(parameter_range, mean_test_score,
       label = "Cross Validation Score", color = 'g')
    # Creating the plot
    plt.title("Validation Curve with RF Classifier")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig("plots/%s.jpeg" %plotname)
    plt.show()

#svc_m = LinearSVC(random_state=0) #linear gave best, squared_hinge, C = 1

RF_validation_curve(tf_X_train_vader.toarray(),Y_train_vader.to_numpy().ravel(), 'validation_curve_tfidf_vader')
RF_validation_curve(tf_X_train_vader_pca,Y_train_vader.to_numpy().ravel(), 'validation_curve_tfidf_vader_pca')
RF_validation_curve(glove200_X_train_vader,Y_train_vader.to_numpy().ravel(), 'validation_curve_glove200_vader')
RF_validation_curve(google300_X_train_vader,Y_train_vader.to_numpy().ravel(), 'validation_curve_google300_vader')

RF_validation_curve(tf_X_train_sentinet.toarray(),Y_train_sentinet.to_numpy().ravel(), 'validation_curve_tfidf_sentinet')
RF_validation_curve(tf_X_train_sentinet_pca,Y_train_sentinet.to_numpy().ravel(), 'validation_curve_tfidf_sentinet_pca')
RF_validation_curve(glove200_X_train_sentinet,Y_train_sentinet.to_numpy().ravel(), 'validation_curve_glove200_sentinet')
RF_validation_curve(google300_X_train_sentinet,Y_train_sentinet.to_numpy().ravel(), 'validation_curve_google300_sentinet')


#%% Model Functions

def SVC_model(x_train, x_test, y_train, y_test):
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    clf = LinearSVC(random_state=0, dual = False)
    clf.fit(x_train, y_train)
    y_test_pred=clf.predict(x_test)
    report = classification_report(y_test, y_test_pred,output_dict=True)
    return report 

def RandomForest_model(x_train, x_test, y_train, y_test, est):
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    clf = RandomForestClassifier(n_estimators = est, random_state=0)
    clf.fit(x_train, y_train)
    y_test_pred=clf.predict(x_test)
    report = classification_report(y_test, y_test_pred,output_dict=True)
    return report 


# SVM/SVC models 
SVC_tfidf_vader_report = SVC_model(tf_X_train_vader.toarray(), tf_X_test_vader.toarray(), Y_train_vader, Y_test_vader)
SVC_tfidf_vaderpca_report = SVC_model(tf_X_train_vader_pca, tf_X_test_vader_pca, Y_train_vader, Y_test_vader)
SVC_tfidf_sentinet_report = SVC_model(tf_X_train_sentinet.toarray(), tf_X_test_sentinet.toarray(), Y_train_sentinet, Y_test_sentinet)
SVC_tfidf_sentinetpca_report = SVC_model(tf_X_train_sentinet_pca, tf_X_test_sentinet_pca, Y_train_sentinet, Y_test_sentinet)
SVC_glove200_vader_report = SVC_model(glove200_X_train_vader, glove200_X_test_vader, Y_train_vader, Y_test_vader)
SVC_glove200_sentinet_report = SVC_model(glove200_X_train_sentinet, glove200_X_test_sentinet, Y_train_sentinet, Y_test_sentinet)
SVC_google300_vader_report = SVC_model(google300_X_train_vader, google300_X_test_vader, Y_train_vader, Y_test_vader)
SVC_google300_sentinet_report = SVC_model(google300_X_train_sentinet, google300_X_test_sentinet, Y_train_sentinet, Y_test_sentinet)

# RF models with n_estimators determined from validation curves
RandomForest_tfidf_vader_report = RandomForest_model(tf_X_train_vader.toarray(), tf_X_test_vader.toarray(), Y_train_vader, Y_test_vader, 450)
RandomForest_tfidf_vaderpca_report = RandomForest_model(tf_X_train_vader_pca, tf_X_test_vader_pca, Y_train_vader, Y_test_vader, 750)
RandomForest_tfidf_sentinet_report = RandomForest_model(tf_X_train_sentinet.toarray(), tf_X_test_sentinet.toarray(), Y_train_sentinet, Y_test_sentinet, 100)
RandomForest_tfidf_sentinetpca_report = RandomForest_model(tf_X_train_sentinet_pca, tf_X_test_sentinet_pca, Y_train_sentinet, Y_test_sentinet, 100)
RandomForest_glove200_vader_report = RandomForest_model(glove200_X_train_vader, glove200_X_test_vader, Y_train_vader, Y_test_vader, 100)
RandomForest_glove200_sentinet_report = RandomForest_model(glove200_X_train_sentinet, glove200_X_test_sentinet, Y_train_sentinet, Y_test_sentinet, 100)
RandomForest_google300_vader_report = RandomForest_model(google300_X_train_vader, google300_X_test_vader, Y_train_vader, Y_test_vader, 300)
RandomForest_google300_sentinet_report = RandomForest_model(google300_X_train_sentinet, google300_X_test_sentinet, Y_train_sentinet, Y_test_sentinet, 100)

#all models
models = [SVC_tfidf_vader_report, SVC_tfidf_vaderpca_report, 
          SVC_tfidf_sentinet_report, SVC_tfidf_sentinetpca_report,
            SVC_glove200_vader_report, SVC_glove200_sentinet_report, 
            SVC_google300_vader_report, SVC_google300_sentinet_report, 
            RandomForest_tfidf_vader_report, RandomForest_tfidf_vaderpca_report,
            RandomForest_tfidf_sentinet_report, RandomForest_tfidf_sentinetpca_report,
            RandomForest_glove200_vader_report, RandomForest_glove200_sentinet_report, 
            RandomForest_google300_vader_report, RandomForest_google300_sentinet_report]

# all model names 
names = ['SVC_tfidf_vader_report', 'SVC_tfidf_vaderpca_report', 
         'SVC_tfidf_sentinet_report',  'SVC_tfidf_sentinetpca_report', 
         'SVC_glove200_vader_report', 'SVC_glove200_sentinet_report', 
         'SVC_google300_vader_report','SVC_google300_sentinet_report',
         'RandomForest_tfidf_vader_report', 'RandomForest_tfidf_vaderpca_report',
         'RandomForest_tfidf_sentinet_report', 'RandomForest_tfidf_sentinetpca_report',
         'RandomForest_glove200_vader_report', 'RandomForest_glove200_sentinet_report', 
         'RandomForest_google300_vader_report', 'RandomForest_google300_sentinet_report']


# pulling model perfromance from array of models 
avg = []
neg = []
neu = []
pos = []
accuracy = []
macro_avg = []
for i in models:
    avg.append(i['weighted avg'])
    neg.append(i['NEG'])
    neu.append(i['NEU'])
    pos.append(i['POS'])
    accuracy.append(i['accuracy'])
    macro_avg.append(i['macro avg'])
avg = pd.Series(avg)
neg = pd.Series(neg)
neu = pd.Series(neu)
pos = pd.Series(pos)
macro_avg = pd.Series(macro_avg)

# saving weighted results 
weighted_results = pd.DataFrame()   
weighted_results['model'] = names
weighted_results[['model', 'transformation', 'lexicon', 'report']] =  weighted_results['model'].str.split('_',expand=True)
weighted_results['precision'] = avg.apply(lambda score_dict: score_dict['precision'])
weighted_results['recall'] = avg.apply(lambda score_dict: score_dict['recall'])
weighted_results['f1-score'] = avg.apply(lambda score_dict: score_dict['f1-score'])
weighted_results = weighted_results.drop(['report'], axis = 1)
weighted_results.to_pickle("./pickles/weighted_results.pkl")

#saving pos results 
pos_results = pd.DataFrame()   
pos_results['model'] = names
pos_results[['model', 'transformation', 'lexicon', 'report']] =  pos_results['model'].str.split('_',expand=True)
pos_results['precision'] = pos.apply(lambda score_dict: score_dict['precision'])
pos_results['recall'] = pos.apply(lambda score_dict: score_dict['recall'])
pos_results['f1-score'] = pos.apply(lambda score_dict: score_dict['f1-score'])
pos_results = pos_results.drop(['report'], axis = 1)
pos_results.to_pickle("./pickles/pos_results.pkl")

#saving neu results 
neu_results = pd.DataFrame()   
neu_results['model'] = names
neu_results[['model', 'transformation', 'lexicon', 'report']] =  neu_results['model'].str.split('_',expand=True)
neu_results['precision'] = neu.apply(lambda score_dict: score_dict['precision'])
neu_results['recall'] = neu.apply(lambda score_dict: score_dict['recall'])
neu_results['f1-score'] = neu.apply(lambda score_dict: score_dict['f1-score'])
neu_results = neu_results.drop(['report'], axis = 1)
neu_results.to_pickle("./pickles/neu_results.pkl")

#saving neg results 
neg_results = pd.DataFrame()   
neg_results['model'] = names
neg_results[['model', 'transformation', 'lexicon', 'report']] =  neg_results['model'].str.split('_',expand=True)
neg_results['precision'] = neg.apply(lambda score_dict: score_dict['precision'])
neg_results['recall'] = neg.apply(lambda score_dict: score_dict['recall'])
neg_results['f1-score'] = neg.apply(lambda score_dict: score_dict['f1-score'])
neg_results = neg_results.drop(['report'], axis = 1)
neg_results.to_pickle("./pickles/neg_results.pkl")