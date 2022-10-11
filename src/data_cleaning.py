# -*- coding: utf-8 -*-
"""
Created on Wed Oct  10 23:17:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""

import pandas as pd
import os 
import re
from nltk.stem import WordNetLemmatizer
  

  
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


abbr_dict = pd.read_csv('../data/raw/abbreviation.csv', header=None, index_col=0, squeeze=True).to_dict()

df = pd.read_csv('../data/raw/train.csv')
df = df.dropna()



def _lookup_words(text):
    if type(text) == float:
        return text
    text = text.lower()
    text =  re.sub('[^A-Za-z0-9 ]+', '', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    new_words = [] 
    
    for word in words:
        if word.lower() in abbr_dict:
            word = abbr_dict[word.lower()]
        word = lemmatizer.lemmatize(word)
        new_words.append(word)
    
    new_text = " ".join(new_words)
    
    return new_text 

df['q1_cleaned'] = df['question1'].apply(_lookup_words)


df['q2_cleaned'] = df['question1'].apply(_lookup_words)


df = df[['q1_cleaned', 'q2_cleaned']]
#df = df[['q1_cleaned', 'q2_cleaned', 'is_duplicate']]
#df.to_csv("../data/processed/clean_data.csv", index = False)

topic_df = pd.read_feather('../data/processed/train_w_topic_model.feather')

topic_df = topic_df.rename(columns={"q1_cleaned": "q1_cleaned_V1", "q2_cleaned": "q2_cleaned_V2"})

df_merged = pd.concat([topic_df, df], axis = 1).drop(['index','id', 'question1','question2'], axis=1)


print(df_merged.columns)
df_merged.to_feather('../data/processed/train_w_topic_model_w_clean_data.feather')



