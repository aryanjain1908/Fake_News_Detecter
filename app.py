import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask , render_template , request , url_for , redirect


app = Flask(__name__)

@app.route('/is_fake/<news>')
def is_fake(news):
    
    df = pd.read_csv('https://github.com/aryanjain1908/UpGrad-UGC/blob/master/train.csv?raw=True' , index_col=0)
    df = df.dropna()
    
    df['Whole'] = df['title'] + df['text']
    
    labels = df.label
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    
    tfidf_train = tfidf_vectorizer.fit_transform(df['Whole']) 
    tfidf_test=tfidf_vectorizer.transform([news])
    
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train,labels)

    y_pred = pac.predict(tfidf_test)
    
    if y_pred[0] == 0:
        return render_template("index.html" , news = "FAKE NEWS")
    else:
        return render_template("index.html" , news = "REAL NEWS")

    
@app.route('/' , methods=['GET','POST'])
def main():
    if request.method == "POST":
        news = request.form['hero-field']
        return redirect(url_for('is_fake' , news = news))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
