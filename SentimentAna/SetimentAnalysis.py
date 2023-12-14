from keras.models import load_model
import pickle
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd


def decode_sentiment(score,include_neutral=True):
    if include_neutral:
        label = "neutral"
        if score <= 0.4:
            label = "negative"
        elif score >=0.7:
            label = "positive"

        return label
    
    else:
        return "negative" if score<0.5 else "positive"


def predict(text,include_neutral=True):
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=300)
    score=model.predict([x_test])[0]
    label=decode_sentiment(score,include_neutral=include_neutral)
    return {"label":label,"score":float(score),"elapsed_time":time.time()-start_at}


def predict_csv(text,include_neutral=True):
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=300)
    score=model.predict([x_test])[0]
    label=decode_sentiment(score,include_neutral=include_neutral)
    return label

model = load_model("model.h5")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
#print("\n\n\n",predict("yo nigga"))
df=pd.read_csv("Data/TEXT1.csv")
df['Sentiment'] = df["Text"].apply(predict_csv)



import matplotlib.pyplot as plt
print(df['Sentiment'].value_counts())
d=dict(df['Sentiment'].value_counts())
x=['positive','negative','neutral']
y=[]
for i in x:
    y.append(d[i])
plt.bar(x,y)
plt.show()