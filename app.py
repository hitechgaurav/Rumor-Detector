# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:07:40 2020

@author: Gaurav SINGH
"""

import pickle

model = pickle.load(open('final_model.sav', 'rb'))

import numpy as np

text = 'i am gaurav kumar'
prd = model.predict(np.array([text]))

pob= model.predict_proba([text])[0][1]
