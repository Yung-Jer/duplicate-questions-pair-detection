# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:10:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""
# python3 -m spacy download enimport numpy as np
import pandas as pd

import os
from nltk.corpus import stopwords

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

TOTAL_TOPICS = 40
train_df_raw = pd.read_csv('../data/raw/train.csv')
train_df = train_df_raw.dropna()


stop_words = stopwords.words('english')



#convert column to list
q1s=train_df['question1'].tolist()
q2s=train_df['question2'].tolist()

qs = q1s + q2s

