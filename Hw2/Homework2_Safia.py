#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re 
from gensim.models import Word2Vec


# In[3]:


#Read all the files from the directory 
directory_path = 'news/20news-bydate-train/'
directories = os.listdir(directory_path)

corpus = []
for d in directories:
    subpath = os.path.join(directory_path, d)
    subdirs = os.listdir(subpath)
    for file in subdirs:
        filepath = os.path.join(subpath, file)
        with open(filepath, encoding='utf-8', errors='ignore') as infile:
            corpus.append(infile.read())


# In[4]:


# Normalize
corpus_normalized = []
for doc in corpus: 
    doc = re.sub(r'[^A-Za-z\s]', '', doc)
    doc = re.sub(r'[\n\t]', ' ', doc)
    doc = doc.lower()
    corpus_normalized.append(doc)


# In[5]:


# Tokenize 
corpus_tokenized = []
for doc in corpus_normalized:
    corpus_tokenized.append(word_tokenize(doc))


# In[6]:


# Output the preprocessing results as a text file, each line containing a single document
preprocessed_docs = ''
for doc in corpus_tokenized:
    preprocessed_docs += ' '.join(doc) + '\n'

path = '/Users/safia/Desktop/TextAnalytics/homework/homework2/news_preprocessed.txt'
with open(path, 'w') as outfile:
    outfile.write(preprocessed_docs)


# ## Cbow model (sg=0)

# ### Embedding size = 20 

# In[12]:


model_1_cbow = Word2Vec(corpus_tokenized, size= 20, workers=20, window =20, sg = 0)


# In[25]:


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


# ### Embedding size = 60 

# In[7]:


model_2_cbow = Word2Vec(corpus_tokenized, size= 60, workers=20, window =20, sg = 0)


# In[8]:


model_2_cbow.wv.most_similar(positive = "king")


# In[9]:


model_2_cbow.wv.most_similar(positive = "woman")


# In[10]:


model_2_cbow.wv.most_similar(positive = "car")


# In[11]:


model_2_cbow.wv.most_similar(positive = "money")


# In[12]:


model_2_cbow.wv.most_similar(positive = "run")


# In[13]:


model_2_cbow.wv.most_similar(positive = "live")


# In[14]:


model_2_cbow.wv.most_similar(positive = "drink")


# In[15]:


model_2_cbow.wv.most_similar(positive = "understand")


# In[16]:


model_2_cbow.wv.most_similar(positive = "the")


# ## Skipgram model (sg = 1) 

# In[17]:


model_1_sg = Word2Vec(corpus_tokenized, size= 20, workers=20, window =20, sg = 1)


# In[18]:


model_1_sg.wv.most_similar(positive = "king")


# In[19]:


model_1_sg.wv.most_similar(positive = "woman")


# In[20]:


model_1_sg.wv.most_similar(positive = "car")


# In[21]:


model_1_sg.wv.most_similar(positive = "money")


# In[22]:


model_1_sg.wv.most_similar(positive = "run")


# In[23]:


model_1_sg.wv.most_similar(positive = "live")


# In[24]:


model_1_sg.wv.most_similar(positive = "drink")


# In[25]:


model_1_sg.wv.most_similar(positive = "understand")


# In[26]:


model_1_sg.wv.most_similar(positive = "the")


# ### Embedding size = 60 

# In[27]:


model_2_sg = Word2Vec(corpus_tokenized, size= 60, workers=20, window =20, sg = 1)


# In[28]:


model_2_sg.wv.most_similar(positive = "king")


# In[29]:


model_2_sg.wv.most_similar(positive = "woman")


# In[30]:


model_2_sg.wv.most_similar(positive = "car")


# In[31]:


model_2_sg.wv.most_similar(positive = "money")


# In[32]:


model_2_sg.wv.most_similar(positive = "run")


# In[33]:


model_2_sg.wv.most_similar(positive = "live")


# In[34]:


model_2_sg.wv.most_similar(positive = "drink")


# In[35]:


model_2_sg.wv.most_similar(positive = "understand")


# In[36]:


model_2_sg.wv.most_similar(positive = "the")


# In[ ]:




