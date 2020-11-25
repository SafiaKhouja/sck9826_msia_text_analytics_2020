#!/usr/bin/env python
# coding: utf-8

# In[24]:


import re
import eli5
import numpy as np 
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# # References for code: 
# - https://eli5.readthedocs.io/en/latest/tutorials/black-box-text-classifiers.html#
# - https://towardsdatascience.com/adding-interpretability-to-multiclass-text-classification-models-c44864e8a13b
# - https://www.kaggle.com/mlwhiz/interpreting-text-classification-models-with-eli5

# # Data Processing and Modeling Functions

# In[ ]:


def load_large_df():
    ## load all the data
    sub1 = pd.read_csv("twitgen_test.csv")
    sub2 = pd.read_csv("twitgen_train.csv")
    sub3 = pd.read_csv("twitgen_valid.csv")

    # concatenate the data
    df = pd.concat([sub1, sub2, sub3])

    # make a binary column (male = 1, female = 0)
    df["male_bin"] = df["male"]*1
    # select only the text and label column
    df = df[["male_bin", "text"]]
    return df


# In[10]:


def load_small_df():
    # load the data
    df = pd.read_csv("gender-classifier.csv", encoding = "latin1")
    # select only the male and female tweets (not the brand or unknown tweets)
    df = df[(df["gender"] == "male") | (df["gender"] == "female")]
    # make a binary column (male = 1, female = 0)
    df["male_bin"] = pd.get_dummies(df['gender'])["male"]
    # clean the text to remove non word entities
    df["text"] = df["text"].apply(lambda x: re.sub(r'#(\w+)', '', x))
    df["text"] = df["text"].apply(lambda x: re.sub(r'@(\w+)', '', x))
    df["text"] = df["text"].apply(lambda x: re.sub(r'https://(\w+)', '', x))
    df["text"] = df["text"].apply(lambda x: re.sub(r'www.(\w+)', '', x))
    # select only text and label column
    df = df[["male_bin", "text"]]
    return df


# In[5]:


def final_df():
    # Load both datasets and concatenate them
    large = load_large_df()
    small = load_small_df()
    df = pd.concat([large, small])
    # need to reset index since we concatenated data
    df = df.reset_index(drop=True)
    return df


# In[13]:


def train_best_model(X, y):
    # build the model pipeline
    vec = TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))
    lr = LogisticRegressionCV(cv = 3, max_iter = 3000)
    pipe = make_pipeline(vec, lr)
    pipe.fit(X, y)
    return pipe


# In[82]:


def print_report(pipe, X_test, y_test):
    y_actuals = y_test
    y_preds = pipe.predict(X_test)
    report = metrics.classification_report(y_actuals, y_preds)
    print(report)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals, y_preds)))


# # Train and Test Set

# In[83]:


# Train the model on the training and testing data to find test accuracy 
df = final_df()
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["male_bin"], test_size=0.15, random_state=20)
pipe_train = train_best_model(X_train, y_train)
# print accuracy metrics 
print_report(pipe_train, X_test, y_test)


# In[85]:


# make a dataframe to ask others to predict gender with for comparison with the algorithm 
test = pd.concat([X_test, y_test], axis = 1) 
y_preds = pipe_train.predict(test['text'])
test['predicted_label'] = y_preds


# In[87]:


sample = test[0:100]
metrics.accuracy_score(test["male_bin"], test["predicted_label"])


# In[89]:


# make a dataframe to ask others to predict gender with for comparison with the algorithm 
df_test = df 
df_test = df_test[df_test.index.isin(y_test.index)]
y_preds = pipe_train.predict(df_test['text'])
df_test['predicted_label'] = y_preds


# In[181]:


sample = df_test[0:100]
metrics.accuracy_score(df_test["male_bin"], df_test["predicted_label"])


# In[121]:


sample.to_csv("sample.csv", encoding = "utf-8")


# Results from my family

# In[175]:


zayne = pd.read_csv("sample_zayne.csv")
metrics.accuracy_score(zayne.iloc[:, [1]], zayne["predicted_label"])


# In[176]:


adel = pd.read_csv("sample_adel.csv")
metrics.accuracy_score(adel["male_bin"], adel["predicted_label"])


# In[177]:


maria = pd.read_csv("sample_maria.csv")
metrics.accuracy_score(adel["male_bin"], adel["predicted_label"])


# In[180]:


safia = pd.read_csv("saf_sample.csv")
metrics.accuracy_score(safia["male_bin"], safia["predicted_label"])


# In[182]:


(.62 + .53 + .53 + .57)/4


# # All data

# In[22]:


# Train the model on all the training data available 
df = final_df()
X = df["text"]
y = df["male_bin"]
# Dont use the function above because we want to have all parts of the model exposed for interpretation steps 
vec = TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))
lr = LogisticRegressionCV(cv = 3, max_iter = 3000)
pipe = make_pipeline(vec, lr)
pipe.fit(X, y)


# In[31]:


eli5.show_weights(lr, vec=vec, top=15)


# In[208]:


y_preds = pipe.predict(df_test['text'])
df_test['predicted_label'] = y_preds
eli5.show_prediction(lr, df_test['text'].values[338], vec=vec)


# In[ ]:




