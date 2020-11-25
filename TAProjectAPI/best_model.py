import pandas as pd
import re
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib

def load_large_df():
    ## load all the data
    sub1 = pd.read_csv("data/twitgen_test.csv")
    sub2 = pd.read_csv("data/twitgen_train.csv")
    sub3 = pd.read_csv("data/twitgen_valid.csv")

    # concatenate the data
    df = pd.concat([sub1, sub2, sub3])

    # make a binary column (male = 1, female = 0)
    df["male_bin"] = df["male"]*1
    # select only the text and label column
    df = df[["male_bin", "text"]]
    return df

def load_small_df():
    # load the data
    df = pd.read_csv("data/gender-classifier.csv", encoding = "latin1")
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

def final_df():
    # Load both datasets and concatenate them
    large = load_large_df()
    small = load_small_df()
    df = pd.concat([large, small])
    # need to reset index since we concatenated data
    df = df.reset_index(drop=True)
    return df

def train_best_model():
    # load the final dataframe
    df = final_df()
    X_train = df["text"]
    y_train = df["male_bin"]
    # build the model pipeline
    vec = TfidfVectorizer(strip_accents = "unicode", stop_words='english', ngram_range=(1, 1))
    lr = LogisticRegressionCV(cv = 3, max_iter = 3000)
    pipe = make_pipeline(vec, lr)
    pipe.fit(X_train, y_train)
    return pipe

if __name__ == '__main__':
    pipeline = train_best_model()
    joblib.dump(pipeline, 'pipeline.pkl')

