#!/usr/bin/env python
# coding: utf-8

# In[21]:


import json
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import gensim
import sklearn.datasets
import sklearn.linear_model 


# In[2]:


# open the subsetted data
with open("yelp_subset.json", 'r') as infile:
    sub = json.load(infile)


# In[4]:


# a very simple tokenizer that splits on white space and gets rid of some punctuation
def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&?!#@$%^&*:;!').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens


# In[ ]:


# UNIGRAMS
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


# In[ ]:


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


# In[ ]:


def all_processing(uni, tokens_to_keep):
    # use multiprocessing to more efficiently process the text 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    if uni == True: 
        result = pool.map(process_line_unigrams, sub)
    else: 
        result = pool.map(process_line_unigrams_bigram, sub)
    texts, labels = zip(*result)
    # create a dictionary
    mydict = gensim.corpora.Dictionary(texts)
    # remove words that occur less than 5 times and words that appear in more than 50% of all documents
    # keep_n means keep the n most frequent tokens 
    mydict.filter_extremes(keep_n=tokens_to_keep)
    # create a tf-idf model
    tfidf_model_unigrams = gensim.models.TfidfModel(dictionary=mydict)
    # create a tf-idf bow corpus
    corpus = [mydict.doc2bow(x) for x in texts]
    corpus_tfidf = tfidf_model_unigrams[corpus]
    # save the serialized work for the linear regression 
    gensim.corpora.SvmLightCorpus.serialize("review.liblinear",corpus_tfidf, labels=labels)


# In[ ]:


all_processing(True, 50000)
X,y = sklearn.datasets.load_svmlight_file("review.liblinear")
lr = sklearn.linear_model.LogisticRegression(C=2)
lr.fit(X,y)
logging.info(lr.score(X, y))


# In[ ]:




