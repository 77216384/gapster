![alt text](https://github.com/exchez/gapster/gapster_sm.png "Gapster")

Gapster is an API for submitting text and getting back a fill-in-the-blank style question with multiple choice generation and answer checking.

## Business Understanding
As the growth of online communities based around content providers grows, so does the need to foster quality discussion around the content being provided. What if you could prevent spam and promote discussion quality by simply having users answer a fill-in-the blank question about the article before they can leave a comment? Gapster will provide this service to content providers through a simple and fast API.

## Data Understanding
Reasearch on Question Generation has been ongoing in academia since at least 2002. Microsft Research has made a dataset available of labeled fill-in-the-blank questions that I will use to train a Machine Learning model.

## Data Preparation
I have prepared a pipeline for part of speech (POS) tagging of the dataset and incomming queries. This will allow for metadata about each question to be created and fed into the Machine Learing model.

There will also be tagging and chunking of phrases to find the best potential word or phrase to use for the question.

Finally, I will be trying out 2-3 different text summarization algorithms to determine which sentences will be best for running through the algorithm.

## Modeling
The main Machine Learning model I will be testing on are Gradient Boosted Decision Trees I will also evaluate the performance of a Random Forest model and a model using XGBoost.

## Deployment
My minium viable product will be an API which recieves text and returns a fill-in-the-blank question. Enhancements will be made along the way as end to end project milestones are reached. Once a working API is done, I will then begin work on generating feasible multiple choice questions based on the provided document using word2vec an WordNet Synsets. Also, further work will be done to provide API support for answer checking and the ability to recieving more than one question for an article. 
