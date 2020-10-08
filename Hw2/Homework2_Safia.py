#!/usr/bin/env python
# coding: utf-8

# In[124]:


import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re 
from gensim.models import Word2Vec


# In[21]:


#Read all the files from the directory 
directory_path = '/Users/safia/Desktop/TextAnalytics/homework/homework2/news/20news-bydate-train/'
directories = os.listdir(directory_path)

corpus = []
for d in directories:
    subpath = os.path.join(directory_path, d)
    subdirs = os.listdir(subpath)
    for file in subdirs:
        filepath = os.path.join(subpath, file)
        with open(filepath, encoding='utf-8', errors='ignore') as infile:
            corpus.append(infile.read())


# In[75]:


# Normalize
corpus_normalized = []
for doc in corpus: 
    doc = re.sub(r'[^A-Za-z\s]', '', doc)
    doc = re.sub(r'[\n\t]', ' ', doc)
    doc = doc.lower()
    corpus_normalized.append(doc)


# In[77]:


# Tokenize 
corpus_tokenized = []
for doc in corpus_normalized:
    corpus_tokenized.append(word_tokenize(doc))


# In[117]:


# Output the preprocessing results as a text file, each line containing a single document
preprocessed_docs = ''
for doc in corpus_tokenized:
    preprocessed_docs += ' '.join(doc) + '\n'

path = '/Users/safia/Desktop/TextAnalytics/homework/homework2/news_preprocessed.txt'
with open(path, 'w') as outfile:
    outfile.write(preprocessed_docs)


# Cbow model (sg=0)

# In[154]:


model_1_cbow = Word2Vec(corpus_tokenized, size= 20, workers=20, window =20, sg = 0)


# In[160]:


model_1_cbow.wv.most_similar(positive = "king")


# In[170]:


model_1_cbow.wv.most_similar(positive = "woman")


# In[173]:


model_1_cbow.wv.most_similar(positive = "car")


# In[174]:


model_1_cbow.wv.most_similar(positive = "money")


# In[176]:


model_1_cbow.wv.most_similar(positive = "run")


# In[178]:


model_1_cbow.wv.most_similar(positive = "live")


# In[179]:


model_1_cbow.wv.most_similar(positive = "drink")


# In[191]:


model_1_cbow.wv.most_similar(positive = "understand")


# In[193]:


model_1_cbow.wv.most_similar(positive = "the")


# Skipgram model (sg = 1) 

# In[161]:


model_1_sg = Word2Vec(corpus_tokenized, size= 10, workers=10, window =10, sg = 1)


# In[162]:


model_1_sg.wv.most_similar(positive = "king")


# In[180]:


model_1_sg.wv.most_similar(positive = "woman")


# In[182]:


model_1_sg.wv.most_similar(positive = "car")


# In[185]:


model_1_sg.wv.most_similar(positive = "money")


# In[186]:


model_1_sg.wv.most_similar(positive = "run")


# In[187]:


model_1_sg.wv.most_similar(positive = "live")


# In[188]:


model_1_sg.wv.most_similar(positive = "drink")


# In[194]:


model_1_sg.wv.most_similar(positive = "understand")


# In[196]:


model_1_sg.wv.most_similar(positive = "the")


# In[ ]:




