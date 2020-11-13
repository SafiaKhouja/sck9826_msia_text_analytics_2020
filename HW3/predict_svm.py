#!/usr/bin/env python
# coding: utf-8

# In[65]:


import json
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import gensim
import sklearn.datasets
import sklearn.svm 


# In[66]:


# open the subsetted data
with open("yelp_subset.json", 'r') as infile:
    sub = json.load(infile)


# In[67]:


# a very simple tokenizer that splits on white space and gets rid of some punctuation
def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&?!#@$%^&*:;!').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens


# In[68]:


# BIGRAMS 
def process_line_unigram_bigram(line):
    # convert the text line to a json object
    json_object = json.loads(line)
    # read and tokenize the text
    text=json_object['text']
    # Unigrams — the same as making tokens
    unigrams=tokenize(text)
    # the bigrams just contatenate 2 adjacent tokens with _ in between    
    bigrams=list(map(lambda x: '_'.join(x), zip(unigrams, unigrams[1:])))    
    # returning a list containing all 1 and 2-grams  
    tokens = unigrams + bigrams
    # read the label and convert to an integer
    label=int(json_object['stars'])
    # return the tokens and the label
    return tokens, label


# In[88]:


def process_line_unigrams(line):
    # convert the text line to a json object
    json_object = json.loads(line)
    # read and tokenize the text
    text=json_object['text']
    # tokenizing is akin to unigrams
    tokens=tokenize(text)
    # read the label and convert to an integer
    label=int(json_object['stars'])
    # return the tokens and the label
    return tokens, label


# In[92]:


def all_processing(uni, tokens_to_keep):
    # use multiprocessing to more efficiently process the text 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    if uni == True: 
        result = pool.map(process_line_unigrams, sub)
    else: 
        result = pool.map(process_line_unigram_bigram, sub)
    texts, labels = zip(*result)
    # create a dictionary
    mydict = gensim.corpora.Dictionary(texts)
    # remove words that occur less than 5 times and words that appear in more than 50% of all documents
    # keep_n means keep the n most frequent tokens 
    mydict.filter_extremes(keep_n=tokens_to_keep)
    # save dictionary
    mydict.save("reviews.dict")


# In[95]:


def make_prediction(review_text):
    #process reviews
    processed = tokenize(review_text)
    # Load dictionary
    mydict = gensim.corpora.Dictionary.load("reviews.dict")   
    print(mydict)
    # create a tf-idf model
    tfidf_model_unigrams = gensim.models.TfidfModel(dictionary=mydict)
    # create a tf-idf bow corpus
    corpus = [mydict.doc2bow(processed)]
    corpus_tfidf = tfidf_model_unigrams[corpus]
    # save the serialized work for the linear regression 
    gensim.corpora.SvmLightCorpus.serialize("prediction.liblinear",corpus_tfidf)
    # fit the model
    X,y = sklearn.datasets.load_svmlight_file("review.liblinear")
    svc = sklearn.svm.LinearSVC(C=2)
    svc.fit(X,y)
    # make the prediction
    X_rev, y_rev = sklearn.datasets.load_svmlight_file("prediction.liblinear")
    print("Prediction: " + svc.predict(X_rev)[0] + "\nConfidence: " + svc.decision_function(X_rev))


# In[97]:


all_processing(True, 75_000)


# In[ ]:


text = "Family friendly pizza spot with cute pinball machines for the kids to play on while they wait for their ‘za"
make_prediction(text)


# In[ ]:

