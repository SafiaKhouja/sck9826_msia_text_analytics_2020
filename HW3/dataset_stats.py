#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# sometimes running this makes my computer freeze
# read the full review dataset 
with open("yelp_academic_dataset_review.json", 'r') as infile:
    all_data = infile.readlines()
# take only the first 500k 
yelp_sub = all_data[0:500000]
# save the subsetted data 
with open("yelp_subset.json", 'w') as outfile:
    json.dump(sub, outfile)


# In[15]:


# open the subsetted data
with open("yelp_subset.json", 'r') as infile:
    sub = json.load(infile)


# In[18]:


# function to process each yelp line 
def process_yelp_line(line):
    # convert the text line to a json object
    json_object = json.loads(line)
    # read and tokenize the text
    text=json_object['text']
    tokens=tokenize(text)
    # read the label and convert to an integer
    label=int(json_object['stars'])
    # return the tokens and the label
    return tokens, label


# In[19]:


# a very simple tokenizer that splits on white space and gets rid of some punctuation
def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&?!#@$%^&*:;!').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens


# In[21]:


# use multiprocessing to more efficiently process the text with the process_yelp_line function above
pool = multiprocessing.Pool(multiprocessing.cpu_count())
result = pool.map(process_yelp_line, sub)
texts, labels = zip(*result)


# In[29]:


print("Number of labels (stars):")
print(len(np.unique(labels)))


# In[30]:


print("Length of text for analysis")
print(len(texts))


# In[39]:


plt.hist(labels, bins = 5)
plt.title("Histogram of labels")


# In[36]:


print("Average word length:")
print(np.mean(list(map(len, sub)))) 


# In[ ]:


plt.hist(list(map(len, sub)))
plt.title("Histogram of number of words per review")

