import pandas as pd

def sentiment(text, pipe):
    # make the prediction from the text
    prediction = pipe.predict(pd.Series(text))
    if prediction[0] == 1:
        gender = "male"
    else:
        gender = "female"
    return gender
