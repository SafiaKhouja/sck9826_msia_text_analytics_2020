#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np 
import eli5
from sklearn import metrics
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
import struct 
from collections import Counter, defaultdict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf


# # References for code: 
# - https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb   
# - https://www.kaggle.com/yekenot/2dcnn-textclassifier
# - https://www.kaggle.com/mlwhiz/learning-text-classification-textcnn

# In[19]:


# Load the large dataset 
def large_df():
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
    # reset the index because we concatenated
    df = df.reset_index(drop=True)
    return df 


# In[20]:


# Load the small dataset 
def small_df():
    # load the data
    df = pd.read_csv("gender-classifier.csv", encoding = "latin1")
    # select only the male and female tweets (not the brand or unknown tweets)
    df = df[(df["gender"] == "male") | (df["gender"] == "female")]
    # make a binary column (male = 1, female = 0)
    df["male_bin"] = pd.get_dummies(df['gender'])["male"]
    df = df[["male_bin", "text"]]
    # clean the text
    df["text"] = df["text"].apply(lambda x: re.sub(r'#(\w+)', '', x))
    df["text"] = df["text"].apply(lambda x: re.sub(r'@(\w+)', '', x))
    df["text"] = df["text"].apply(lambda x: re.sub(r'https://(\w+)', '', x))
    df["text"] = df["text"].apply(lambda x: re.sub(r'www.(\w+)', '', x))
    return df 


# In[21]:


# concatenate the two datasets to make the final dataset
def final_df():
    large = large_df()
    small = small_df()
    df = pd.concat([large, small])
    # need to reset index since we concatinated data
    df = df.reset_index(drop=True)
    return df 


# In[48]:


df = final_df()
# Make training and testing set 
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["male_bin"], test_size=0.15, random_state=15)
# Separate the X and y for cross validation 
X = df["text"]
y = df["male_bin"]
# Verify classes are nearly balanced in the training set 
print(sum(y_train == 1))
print(sum(y_train == 0))


# # Logistic Regression 

# LogisticRegression performs worse than LogisticRegressionCV (bc of L2 regularization)   
# Count Vectorizer with TFIDF is the way to go 

# In[25]:


lr_unigram_no_CV = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                       ("Logreg", LogisticRegression())])
lr_unigram_no_CV_score = cross_val_score(lr_unigram_no_CV, X, y, cv=5).mean()
print("Logistic regression without Cross Validation (count vectorizer + unigrams):" + str(lr_unigram_no_CV_score))


# In[26]:


lr_unigram_countvectorizer = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                       ("Logreg", LogisticRegressionCV())])
lr_unigram_countvectorizer_score = cross_val_score(lr_unigram_countvectorizer, X, y, cv=5).mean()
print("Logistic regression (count vectorizer + unigrams):" + str(lr_unigram_countvectorizer_score))


# In[27]:


lr_unigrambigram_countvectorizer = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))), 
                             ("Logreg", LogisticRegressionCV())])
lr_unigrambigram_countvectorizer_score = cross_val_score(lr_unigrambigram_countvectorizer, X, y, cv=5).mean()
print("Logistic regression (count vectorizer + unigrams/bigrams):" + str(lr_unigrambigram_countvectorizer_score))


# In[28]:


lr_unigram_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                             ("Logreg", LogisticRegressionCV())])
lr_unigram_tfidf_score = cross_val_score(lr_unigram_tfidf, X, y, cv=5).mean()
print("Logistic regression (tfidf + unigrams):" + str(lr_unigram_tfidf_score))


# In[29]:


# used to be 0.586
lr_unigrambigram_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))), 
                             ("Logreg", LogisticRegressionCV())])
lr_unigrambigram_tfidf_score = cross_val_score(lr_unigrambigram_tfidf, X, y, cv=5).mean()
print("Logistic regression (tfidf + unigrams/bigrams):" + str(lr_unigrambigram_tfidf_score))


# # SVM

# In[ ]:


svm_unigram_countvectorizer = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                       ("linear svc", SVC(kernel="linear", C = 1))])
svm_unigram_countvectorizer_score = cross_val_score(svm_unigram_countvectorizer, X_train, y_train, cv=5).mean()
print("SVM (count vectorizer + unigrams):" + str(svm_unigram_countvectorizer_score))


# In[64]:


svm_unigrambigram_countvectorizer = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))), 
                             ("linear svc", SVC(kernel="linear", C = 1))])
svm_unigrambigram_countvectorizer_score = cross_val_score(svm_unigrambigram_countvectorizer, X_train, y_train, cv=5).mean()
print("SVM (count vectorizer + unigrams/bigrams):" + str(svm_unigrambigram_countvectorizer_score))


# In[ ]:


svm_unigram_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                       ("linear svc", SVC(kernel="linear"))])
svm_unigram_tfidf_score = cross_val_score(svm_unigram_tfidf, X_train, y_train, cv=5).mean()
print("SVM (tfidf + unigrams):" + str(svm_unigram_tfidf_score))


# In[ ]:


svm_unigrambigrams_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))), 
                       ("linear svc", SVC(kernel="linear"))])
svm_unigrambigrams_tfidf_score = cross_val_score(svm_unigrambigrams_tfidf, X_train, y_train, cv=5).mean()
print("SVM (tfidf + unigrams/bigrams):" + str(svm_unigrambigrams_tfidf_score))


# # Naive Bayes 

# In[32]:


multi_nb_unigram_countvectorize = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                                            ("multi_nb",  MultinomialNB())])
multi_nb_unigram_countvectorize_score = cross_val_score(multi_nb_unigram_countvectorize, X, y, cv=5).mean()
print("Multinomial NB (count vectorizer + unigrams):" + str(multi_nb_unigram_countvectorize_score))


# In[37]:


multi_nb_unigrambigram_countvectorize = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))), 
                                            ("multi_nb",  MultinomialNB())])
multi_nb_unigrambigram_countvectorize_score = cross_val_score(multi_nb_unigrambigram_countvectorize, X, y, cv=5).mean()
print("Multinomial NB (count vectorizer + unigrams/bigrams):" + str(multi_nb_unigrambigram_countvectorize_score))


# In[33]:


multi_nb_unigram_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                                   ("multi_nb",  MultinomialNB())])
multi_nb_unigram_tfidf_score = cross_val_score(multi_nb_unigram_tfidf, X, y, cv=5).mean()
print("Multinomial NB (tfidf + unigrams):" + str(multi_nb_unigram_tfidf_score))


# In[35]:


bern_nb_unigram_countvectorize = Pipeline([("count_vectorizer", CountVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                                           ("bern_nb",  BernoulliNB())])
bern_nb_unigram_countvectorize_score = cross_val_score(bern_nb_unigram_countvectorize, X, y, cv=5).mean()
print("Bernouli NB (count vectorize + unigrams):" + str(bern_nb_unigram_countvectorize_score))


# In[36]:


bern_nb_unigram_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))), 
                                           ("bern_nb",  BernoulliNB())])
bern_nb_unigram_tfidf_score = cross_val_score(bern_nb_unigram_tfidf, X, y, cv=5).mean()
print("Bernouli NB (tfidf + unigrams):" + str(bern_nb_unigram_tfidf_score))


# In[38]:


bern_nb_unigrambigram_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 2))), 
                                           ("bern_nb",  BernoulliNB())])
bern_nb_unigrambigram_tfidf_score = cross_val_score(bern_nb_unigrambigram_tfidf, X, y, cv=5).mean()
print("Bernouli NB (tfidf + unigrams/bigrams):" + str(bern_nb_unigrambigram_tfidf_score))


# # Word2Vec

# In[41]:


# read the GloVe file for word embeddings
encoding = 'utf-8'
GLOVE_6B_50D_PATH = "glove.6B.50d.txt"
import numpy as np
with open(GLOVE_6B_50D_PATH, "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
               for line in lines}


# In[50]:


# read the GloVe file line by line and only save vectors that correspond to words from the training set 
glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums


# In[51]:


# train word2vec on all the texts - both training and test set
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


# In[52]:


# tfidf embedding vectorizer for word embeddings 
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# In[53]:


lr_glove = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("Logistic regression", LogisticRegressionCV())])
lr_glove_score = cross_val_score(lr_glove, X, y, cv=5).mean()
print("Logistic regression (glove embeddings):" + str(lr_glove_score))


# In[55]:


lr_w2v = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("Logistic regression", LogisticRegressionCV())])
lr_w2v_score = cross_val_score(lr_w2v, X, y, cv=5).mean()
print("Logistic regression (w2v embeddings):" + str(lr_w2v_score))


# In[60]:


extratrees_glove = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
extratrees_glove_score = cross_val_score(extratrees_glove, X, y, cv=5).mean()
print("Extra Trees (glove embeddings):" + str(extratrees_glove_score))


# In[61]:


extratrees_w2v = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
extratrees_w2v_score = cross_val_score(extratrees_w2v, X, y, cv=5).mean()
print("Extra Trees (w2v embeddings):" + str(extratrees_w2v_score))


# In[59]:


multi_nb_glove = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)), 
                        ("multinomial nb",  BernoulliNB())])
multi_nb_glove_score = cross_val_score(multi_nb_glove, X, y, cv=5).mean()
print("Multinomial NB (glove embeddings):" + str(multi_nb_glove_score))


# # All results!
# Except for neural network model 

# In[62]:


print("LOGISTIC REGRESSION")
print("Logistic regression without Cross Validation (count vectorizer + unigrams):" + str(lr_unigram_no_CV_score))
print("Logistic regression (count vectorizer + unigrams):" + str(lr_unigram_countvectorizer_score))
print("Logistic regression (count vectorizer + unigrams/bigrams):" + str(lr_unigrambigram_countvectorizer_score))
print("Logistic regression (tfidf + unigrams):" + str(lr_unigram_tfidf_score))
print("Logistic regression (tfidf + unigrams/bigrams):" + str(lr_unigrambigram_tfidf_score))
print("\n")
print("SUPPORT VECTOR MACHINES")
print("SVM (count vectorizer + unigrams):" + str(svm_unigram_countvectorizer_score))
print("SVM (count vectorizer + unigrams/bigrams):" + str(svm_unigrambigram_countvectorizer_score))
print("SVM (tfidf + unigrams):" + str(svm_unigram_tfidf_score))
print("SVM (tfidf + unigrams/bigrams):" + str(svm_unigrambigrams_tfidf_score))
print("\n")
print("NAIVE BAYES")
print("Multinomial NB (count vectorizer + unigrams):" + str(multi_nb_unigram_countvectorize_score))
print("Multinomial NB (count vectorizer + unigrams/bigrams):" + str(multi_nb_unigrambigram_countvectorize_score))
print("Multinomial NB (tfidf + unigrams):" + str(multi_nb_unigram_tfidf_score))
print("Bernouli NB (count vectorize + unigrams):" + str(bern_nb_unigram_countvectorize_score))
print("Bernouli NB (tfidf + unigrams):" + str(bern_nb_unigram_tfidf_score))
print("Bernouli NB (tfidf + unigrams/bigrams):" + str(bern_nb_unigrambigram_tfidf_score))
print("\n")
print("\n")
print("WORD EMBEDDINGs")
print("Logistic regression (glove embeddings):" + str(lr_glove_score))
print("Logistic regression (w2v embeddings):" + str(lr_w2v_score))
print("Extra Trees (glove embeddings):" + str(extratrees_glove_score))
print("Extra Trees (w2v embeddings):" + str(extratrees_w2v_score))
print("Multinomial NB (glove embeddings):" + str(multi_nb_glove_score))



# # Neural Network

# In[ ]:


max_features = 44359 # max number of words for dictionar
maxlen = 72 
embed_size = 300


# In[ ]:


# Loading the data
def load_and_prec():
    ## split to train and val
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    ## fill missing values
    train_X = train_df["text"].values
    val_X = val_df["text"].values
    # tokenize text
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    ## Pad the sentences with 0 to achieve constant length 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    ## Get the target values
    train_y = train_df['male_bin'].values
    val_y = val_df['male_bin'].values
    # encoder  
    encoder = LabelEncoder()
    encoder.fit(train_y)
    train_y = encoder.transform(train_y)
    val_y = encoder.transform(val_y)
    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    #shuffle data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    return train_X, val_X,  train_y, val_y, val_df, tokenizer, encoder


# In[ ]:


train_X, val_X,  train_y, val_y, val_df,tokenizer , encoder = load_and_prec()


# In[ ]:


# Word 2 vec Embedding
def load_glove(word_index):
    #create an embedding matrix that keeps only the word2vec for words which are in our word_index
    EMBEDDING_FILE = 'glove.6B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]
    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector  
    return embedding_matrix


# In[ ]:


embedding_matrix = load_glove(tokenizer.word_index)


# In[ ]:


# load the embedding matrix if it has already been generated 
embedding_matrix = np.load('embedding_matrix.csv.npy')


# In[ ]:


def model_cnn(embedding_matrix):
    filter_sizes = [1,2, 3, 4]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(2, activation="softmax")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# In[ ]:


def model_cnn(embedding_matrix):
    filter_sizes = [1,2, 3, 4]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(2, activation="softmax")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# In[ ]:


model = model_cnn(embedding_matrix)
model.summary()


# In[ ]:


def train(model, epochs=5):
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y),callbacks=callbacks)
    model.load_weights(filepath)

train(model, epochs = 10)


# In[ ]:


def print_report_nn(model):
    a=np.argmax(val_y, axis = 1)
    y_actuals = encoder.inverse_transform(a)
    y_preds = model.predict([val_X], batch_size=1024, verbose=0)
    prediction_ = np.argmax(y_preds, axis = 1)
    prediction_ = encoder.inverse_transform(prediction_)
    report = metrics.classification_report(y_actuals, prediction_)
    print(report)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals, prediction_)))

print_report_nn(model)

