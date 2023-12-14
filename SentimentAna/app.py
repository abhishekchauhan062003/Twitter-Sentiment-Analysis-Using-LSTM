from flask import Flask,render_template,request
from keras.models import load_model
import pickle
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd




app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def calculate():
    if(request.method=="POST"):
        try:

            text=str(request.form['text'])
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



            model = load_model("model.h5")
            with open('tokenizer.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)
            result=predict(text)
            if(result["label"]=="positive"):
                path = "static/Happy_Emoji.webp"
            elif(result["label"]=="neutral"):
                path = "static/Neutral-Face.png"
            else:
                path = "static/Sad_Emoji.png"

            return render_template("abhishek2.html",label=result["label"].upper(),score=result["score"]*100,path=path)
        except:
            f = request.files['file']
            f.save(f.filename)

            model = load_model("model.h5")
            with open('tokenizer.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)

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
            

            def predict_csv(text,include_neutral=True):
                start_at = time.time()
                x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=300)
                score=model.predict([x_test])[0]
                label=decode_sentiment(score,include_neutral=include_neutral)
                return label
            

            df=pd.read_csv(f.filename)
            df['Sentiment'] = df["Text"].apply(predict_csv)
            import matplotlib.pyplot as plt
            print(df['Sentiment'].value_counts())
            d=dict(df['Sentiment'].value_counts())
            x=['positive','negative','neutral']
            color=['green','red','yellow']
            y=[]
            for i in x:
                y.append(d[i])
            plt.xlabel("Sentiment")
            plt.ylabel("Occurence")
            plt.bar(x,y,color=color)
            plt.savefig("static/Graph.png")
            return render_template("abhishek3.html",label = df['Sentiment'].value_counts().idxmax().upper())











@app.route("/output")
def output():
    return "<h1>OUTPUT</h1>"


if __name__=="__main__":
    app.run(debug=True,port=5500)