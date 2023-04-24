**SentiWordNet and VADER: Comparative Analysis of the Efficacy of Hybrid Sentiment Analysis Models**

This repository contains the code and data for my senior thesis, which explores and compares different hybrid sentiment analysis procedures using two polarity lexicons: VADER and SentiWordNet. The report can be accessed and downloaded from the `Report` folder. 

*Description*

In this thesis, I conducted an exploratory analysis of different hybrid sentiment analysis procedures to determine which method yields the most accurate results when using the VADER and SentiWordNet polarity lexicons. The analysis was conducted using Python, and the code is available in this repository.

*Files*

The following python files are included in this repository:

1. `dataload.py`: Loads data scraped for this project and cleans/precoesses it using the procedures outlined in the report. 
2. `taggingscript.py`: Using a connection to the Stanford NLP server (details in file), tags and lemmatizes each word in a tweet with POS tag.
3. `sentiword_sentiment.py`: Assigns sentiment values using SentiWordNet polarity lexicon.
4. `VADER_sentiment.py`: Assigns sentiment values using VADER polarity lexicon.
5. `model_prep.py`: Splits data for each lexicon using distribition aware procedure. Applies all numerical feature map transformations: TF-IDF, TF-IDF with SVD, Google 300 Word2Vec, GloVe 200 Word2Vec.
6. `modeling_master.py`: Loads all data, checks parameter tuning using validation curves, and collects results from models. 

The data currenlty in this repository will be overwritten on your local decive when running these files. Each file is built to be able to run independently, given the required data was collected at some point in time. The `testing_phase_files` folder contains python files and data saved from the exploratory phase of my research. For all purposes of the study, this folder can be completely ignored. 

*Usage*

To use the code in this repository, follow these steps:

Clone the repository to your local machine. 
Install the required packages imported at the top of each file.
Open the files in your preferred Python IDE. 
Run the files in order to execute the code. 

*Results*

The results of my analysis can be found in my senior thesis paper, which is included in this repository in the `Report` folder. The code in this repository can be used to replicate the analysis and generate similar results.

*Contact*

If you have any questions about this repository or my senior thesis, please contact me at orenfro23@cmc.edu.
