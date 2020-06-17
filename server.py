# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:55:53 2020

@author: Gaurav SINGH
"""

import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify, render_template

def input_data(title, text):
    import pandas as pd
    from keras.preprocessing import sequence
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    heading_inp = pd.Series([title.split()])
    content_inp = pd.Series([text.split()])
    df_inp = heading_inp.map(str)+" "+content_inp.map(str)
    tokenizer = Tokenizer(num_words = 4000, lower = False , split = " ")
    tokenizer.fit_on_texts(df_inp.values)
    X = tokenizer.texts_to_sequences(df_inp.values)
    X = pad_sequences(X)
    max_length = 1500
    X_input = sequence.pad_sequences(X, max_length)
    
    return X_input


app = Flask(__name__)

import pickle

model = pickle.load(open('rumor_model.sav', 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/second')

def second():
    return render_template('second.html')

@app.route('/senddata', methods = ['POST'])
def senddata():
    title = request.form['title']
    content = request.form['content']
    #check = input_data(title, content)
    predicitons = model.predict(np.array([content]))
    probab = (model.predict_proba(np.array([content]))[0][1]*100).round(3)
    temp = ""
    signal = ""
    output = True if probab>52 else False
    if output == True:
        temp = ' Given above data is Hard Fact / True with probability of '+str(probab)+" %."
        return render_template('second.html',flex_true = temp)
    else:
        temp = ' Given above data is Rumor / Fake with probabilty of '+str(probab)+" %."
        return render_template('second.html',flex_false = temp)
        

if __name__ == '__main__':
    app.run(use_reloader = False, debug=True, threaded = False)
    
    