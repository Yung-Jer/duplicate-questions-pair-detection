# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:17:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""
import pandas as pd
import topic_modeling as tm
import build_startingword_lcs as sw
import time
import Levenshtein
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split

# Variables Setting 
verbose = True
TOTAL_TOPICS = 20

def levenshtein(row):
    s1 = row['question1']
    s2 = row['question2']
    return Levenshtein.ratio(s1, s2)

def trim_sentence(sent):
    try:
        sent = sent.split()
        sent = sent[:128]
        return " ".join(sent)
    except:
        return sent

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['q1_cleaned'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['q2_cleaned'].split(" ")))    
    return len(w1 & w2)

def ngram(tokens, n):
    grams =[tokens[i:i+n] for i in range(len(tokens)-(n-1))]
    return grams

def jaccard_similarity(row):
    """
    Derives the Jaccard similarity
    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    sentence_gram1 = row['q1_cleaned']
    sentence_gram2 = row['q2_cleaned']
    grams1 = ngram(sentence_gram1, 5)
    grams2 = ngram(sentence_gram2, 5)
    intersection = set(grams1).intersection(set(grams2))
    union = set(grams1).union(set(grams2))
    if len(union) == 0: # To prevent division by 0
        return 0
    return float(len(intersection))/float(len(union))

if __name__ ==  '__main__':
    tic = time.time()
    # Read data
    df = pd.read_feather('../data/processed/full_clean.feather')
    
    df['q1_trimmed'] = df['q1_cleaned'].apply(lambda x: trim_sentence(x))
    df['q2_trimmed'] = df['q2_cleaned'].apply(lambda x: trim_sentence(x))
    
    df['Levenshtein'] = df.apply(levenshtein, axis=1)
    df['jaccard_dist'] = df.apply(jaccard_similarity, axis=1)
    df['common_words'] = df.apply(common_words, axis=1)
    df['common_ratio'] = df.apply(lambda row: row['common_words'] / (len(row['q1_cleaned']) + len(row['q2_cleaned'])), axis=1)
    df['length_diff'] = df.question1.apply(lambda x: len(str(x))) - df.question2.apply(lambda x: len(str(x)))
    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_wratio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    # Build Longest Common Substring and Sub Sequence as a new feature
    df = sw.build_lcs(df, verbose)
    
    # Topic Modeling
    
    
    # Split dataset into train/validation/test sets
    # We want to split the data in 80:10:10 for train:valid:test dataset
    train_size=0.8

    X = df.drop(['is_duplicate'],  axis=1)
    y = df['is_duplicate']
    
    # Split dataset into training and remaining (val+test) set first
    X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8, stratify=y)
    
    # Split remaining set equally 10:10
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5,stratify=y_rem)
    
    df.reset_index().to_feather('../data/processed/full_dataset.feather')
    
    # Join the x and y back together column-wise, and output
    train_set = pd.concat([X_train, y_train], axis=1)
    val_set = pd.concat([X_valid, y_valid], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    train_set.reset_index().to_feather('../data/processed/train_dataset.feather')
    val_set.reset_index().to_feather('../data/processed/validation_dataset.feather')
    test_set.reset_index().to_feather('..//data/processed/test_dataset.feather')
    
    # End
    toc = time.time()
    time_taken = toc - tic
    print("Time Taken: " + time.strftime('%H:%M:%S', time.gmtime(time_taken)))
