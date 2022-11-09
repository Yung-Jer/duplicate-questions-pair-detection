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
from pyemd import emd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock,canberra, euclidean, minkowski
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# Variables Setting 
verbose = True
TOTAL_TOPICS = 25

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

def same_starting(q1, q2):
    q1 = q1.split()
    q2 = q2.split()
    
    if q1[0] == q2[0]:
        return 1
    return 0

def same_ending(q1, q2):
    q1 = q1.split()
    q2 = q2.split()
    
    if q1[-1] == q2[-1]:
        return 1
    return 0

def get_wmdistance(series):
    return model.wmdistance(series['q1_cleaned'], series['q2_cleaned'])

def avg_w2v(list_of_sent,model,d):
    sent_vectors = []
    for sent in list_of_sent: 
        doc = [word for word in sent if word in model]
        if doc:
            sent_vec = np.mean(model[doc],axis=0)
        else:
            sent_vec = np.zeros(d)
        sent_vectors.append(sent_vec)
    return sent_vectors

def vectorize(sent):
    return model.infer_vector(nltk.word_tokenize(sent))

def get_TFID_diff(df):
    wordbag = pd.concat([df["question1"], df["question2"]], axis = 0)
    tfidf = TfidfVectorizer(analyzer = "word")
    tfidf.fit(wordbag)
    tfidf_q1 = tfidf.transform(df["question1"])
    tfidf_q2 = tfidf.transform(df["question2"])
    diff = tfidf_q1 - tfidf_q2
    diff_tfidf_L1 = np.sum(np.abs(diff), axis = 1) 
    diff_tfidf_L2 = np.sum(diff.multiply(diff), axis = 1)
    diff_tfidf_L1_norm = 2 * np.array(np.sum(np.abs(diff), axis = 1)) / df[['total_length']].values
    diff_tfidf_L2_norm= 2 * np.array(np.sum(diff.multiply(diff), axis = 1)) / df[['total_length']].values
    df["diff_tfidf_L1"] = diff_tfidf_L1
    df["diff_tfidf_L2"] = diff_tfidf_L2
    df["diff_tfidf_L1_norm"] = diff_tfidf_L1_norm
    df["diff_tfidf_L2_norm"] = diff_tfidf_L2_norm
    return df

def get_sentiment(row):
    q1_sen = sentiment_analyzer.polarity_scores(row["q1_cleaned"])
    q2_sen = sentiment_analyzer.polarity_scores(row["q2_cleaned"])
    diff_neg = np.abs(q1_sen["neg"] - q2_sen["neg"])
    diff_neu = np.abs(q1_sen["neu"] - q2_sen["neu"])
    diff_pos = np.abs(q1_sen["pos"] - q2_sen["pos"])
    diff_com = np.abs(q1_sen["compound"] - q2_sen["compound"])
    return pd.Series([diff_neg, diff_neu, diff_pos, diff_com])

if __name__ ==  '__main__':
    tic = time.time()
    # Read data
    df = pd.read_feather('../data/processed/full_clean.feather')
    
    df['q1_trimmed'] = df['q1_cleaned'].apply(lambda x: trim_sentence(x))
    df['q2_trimmed'] = df['q2_cleaned'].apply(lambda x: trim_sentence(x))
    df['same_question'] = (df['question1'] == df['question2']).astype(int)
    df['freq_q1']=df.groupby('q1_cleaned')['q1_cleaned'].transform('count') # Frequency of question 1 in entire dataset
    df['freq_q2']=df.groupby('q2_cleaned')['q2_cleaned'].transform('count') # Frequency of question 2 in entire dataset
    df['freq_q1+q2'] = df['freq_q1']+df['freq_q2'] # Sum of frequency of question 1 and question 2
    df['freq_q1-q2'] = df['freq_q1']-df['freq_q2'] # Difference of frequency in question 1 and question 2
    df['q1_question_mark_count'] = df.apply(lambda x: x['question1'].count('?'), axis=1)
    df['q2_question_mark_count'] = df.apply(lambda x: x['question2'].count('?'), axis=1)
    df['question_mark_count_diff'] = df['q1_question_mark_count'] - df['q2_question_mark_count']
    df['Levenshtein'] = df.apply(levenshtein, axis=1)
    df['jaccard_dist'] = df.apply(jaccard_similarity, axis=1)
    df['common_words'] = df.apply(common_words, axis=1)
    df['common_ratio'] = df.apply(lambda row: row['common_words'] / (len(row['q1_cleaned']) + len(row['q2_cleaned'])), axis=1) #Common words / Total Words
    df['length_diff'] = df.question1.apply(lambda x: len(str(x))) - df.question2.apply(lambda x: len(str(x)))
    
    df['total_length'] =df.q1_cleaned.apply(lambda x: len(str(x))) + df.q2_cleaned.apply(lambda x: len(str(x)))
    df["length_diff_rate"] = 2 * abs(df['length_diff']) / (df['total_length'])
    
    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_wratio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    df["same_starting"] = df.apply(lambda x: same_starting(x.question1, x.question2), axis=1) # if first word is same, 1 else 0
    df["same_ending"] = df.apply(lambda x: same_ending(x.question1, x.question2), axis=1) # if last word is same, 1 else 0

    # Build Longest Common Substring and Sub Sequence as a new feature
    df = sw.build_lcs(df, verbose)
    
    # Topic Modeling
    data = tm.prep_data_for_topic_modeling(df)
    train = pd.DataFrame({'text':data})
    train_corpus, train_id2word, bigram_train = tm.get_corpus(train)
    
    lda_model = tm.build_topic_model(TOTAL_TOPICS, train_corpus, train_id2word, verbose)
    doc_lda = lda_model[train_corpus]
    pd.set_option('display.max_columns', None)  
    print(tm.get_lda_topics(lda_model, TOTAL_TOPICS))
    
    topics={0:'Social Media/Gadget/Email',1:'Self-help/Learn/Business',2:'Purpose/Energy',3:'Language/Relationship',4:'Food/Health',5:'Interview/Difference/Drug',6:'Year/New/Stock/Company', 
            7:'Job/College/University', 8: 'India/Government/China', 9: 'English/Law/Writing', 10: 'Money/Bank/Online', 11: 'Relationship/Girl/Guy/People/Life',
            12: 'Politics/Trump/Election', 13: 'Assessment/Word/Home', 14: 'Country/Car/Show/Television', 15: 'Free/Ocatopm/Software/Website', 16: 'Engine/Password/Search',
            17: 'Long/Review/Work/Compare', 18: 'Best/Way/Visit', 19: 'Lose/Weight/Time/Travel/Salary', 20: 'Quora/Question/Google/Answer', 21: 'Problem/Increase', 22: 'Sex/Woman/Man',
            23: 'Movie/Video Game/Youtube', 24: 'United States/Day'}
    
    lemm_dfq1 = pd.DataFrame({'Text':bigram_train[0:404287]})
    lemm_dfq2 = pd.DataFrame({'Text':bigram_train[404287:]})

    df['q1_cleaned_t'] = lemm_dfq1['Text']
    df['q2_cleaned_t'] = lemm_dfq2['Text']

    df['q1_cleaned_t'].explode().dropna().groupby(level=0).agg(list)
    df['q2_cleaned_t'].explode().dropna().groupby(level=0).agg(list)

    df.loc[:, 'q1_topic']=df['q1_cleaned_t'].apply(lambda x: tm.get_topic(x, topics) if type(x)==list else 'None')
    df.loc[:, 'q2_topic']=df['q2_cleaned_t'].apply(lambda x: tm.get_topic(x, topics) if type(x)==list else 'None')

    df['same_topic'] = (df['q1_topic'] == df['q2_topic']).astype(int)
    #Topic Modelling OHE
    q1_topic = pd.get_dummies(df.q1_topic, prefix='q1')
    q2_topic = pd.get_dummies(df.q2_topic, prefix='q2')
    frames = [df, q1_topic, q2_topic]
    df = pd.concat(frames, axis = 1)
    
    
    #Build Distance Feature
    model = KeyedVectors.load_word2vec_format("../data/raw/glove_vectors.txt", binary=False, limit=100000)
    q1_list = []
    q2_list = []
    for s in df.q1_trimmed.values:
        q1_list.append(s.split())
    for s in df.q2_trimmed.values:
        q2_list.append(s.split())
    avgw2v_q1 = avg_w2v(q1_list,model,300)
    avgw2v_q2 = avg_w2v(q2_list,model,300)
    df_avgw2v = pd.DataFrame()
    df_avgw2v['q1_vec'] = list(avgw2v_q1)
    df_avgw2v['q2_vec'] = list(avgw2v_q2)
    df_q1 = pd.DataFrame(df_avgw2v.q1_vec.values.tolist())
    df_q2 = pd.DataFrame(df_avgw2v.q2_vec.values.tolist())

    df['wmdistance'] = df.apply(lambda row: get_wmdistance(row[['q1_trimmed', 'q2_trimmed']]) , axis=1)
    df['dist_cosine'] = [cosine(x, y) for (x, y) in zip(avgw2v_q1,avgw2v_q2)]
    df['dist_cityblock'] = [cityblock(x, y) for (x, y) in zip(avgw2v_q1,avgw2v_q2)]
    df['dist_canberra'] = [canberra(x, y) for (x, y) in zip(avgw2v_q1,avgw2v_q2)]
    df['dist_euclidean'] = [euclidean(x, y) for (x, y) in zip(avgw2v_q1,avgw2v_q2)]
    df['dist_minkowski'] = [minkowski(x, y) for (x, y) in zip(avgw2v_q1,avgw2v_q2)]
    df.dist_cosine = df.dist_cosine.fillna(0)

    #Build Work2Vec Feature
    all_words = [nltk.word_tokenize(sent) for sent in df['q1_cleaned']]
    all_words = all_words + [nltk.word_tokenize(sent) for sent in df['q2_cleaned']]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_words)]

    #1 Vector
    model = Doc2Vec(documents, vector_size=1, min_count=2, workers=10)
    df['q1_word_to_vec'] = df['q1_cleaned'].apply(vectorize).apply(lambda x: x[0])
    df['q2_word_to_vec'] = df['q2_cleaned'].apply(vectorize).apply(lambda x: x[0])
    #5 Vector
    model = Doc2Vec(documents, vector_size=5, min_count=2, workers=10)
    df['q1_vector'] = df['q1_cleaned'].apply(vectorize)
    df['q2_vector'] = df['q2_cleaned'].apply(vectorize)
    df[['q1_vec_0', 'q1_vec_1', 'q1_vec_2','q1_vec_3','q1_vec_4']] = pd.DataFrame(df['q1_vector'].tolist())
    df[['q2_vec_0', 'q2_vec_1', 'q2_vec_2','q2_vec_3','q2_vec_4']] = pd.DataFrame(df['q2_vector'].tolist())
    
    #Build Sentiment Feature
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = df.apply(get_sentiment, axis = 1)
    sentiment.columns = ["diff_sen_negative", "diff_sen_neutral", "diff_sen_positive", "diff_sen_compound"]
    df = pd.concat([df, sentiment], axis = 1)

    #Build TFID Diff Feature
    df = get_TFID_diff(df)
    
    '''
    # Rearrange Columns
    df = df[['qid1', 'qid2', 'question1','question2', 'q1_cleaned', 'q2_cleaned',
           'q1_trimmed', 'q2_trimmed', 'q1_start', 'q2_start', 'q1_topic','q2_topic', 'length_diff','same_question',
           'lc_substring', 'lc_subsequence', 'jaccard_dist', 'common_words',
           'common_ratio', 'levenshtein', 'fuzz_qratio', 'fuzz_wratio',
           'q2_question_mark_count', 'q1_question_mark_count','question_mark_count_diff',
           'freq_q1+q2','freq_q1-q2','same_topic','same_starting','same_ending', 'wmdistance',
           'dist_cosine', 'dist_cityblock' , 'dist_canberra', 'dist_euclidean', 'dist_minkowski',
           'is_duplicate']]
    '''
    df.insert(len(df.columns)-1, 'is_duplicate', df.pop('is_duplicate'))
    df.fillna(0, inplace = True)
    # Output full dataset
    df.reset_index().to_feather('../data/processed/full_dataset.feather')
    # Split dataset into train/validation/test sets
    # We want to split the data in 80:10:10 for train:valid:test dataset
    train_size=0.8

    X = df.drop(['is_duplicate'],  axis=1)
    y = df['is_duplicate']
    
    # Split dataset into training and remaining (val+test) set first
    X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8, stratify=y, random_state = 42)
    
    # Split remaining set equally 10:10
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5,stratify=y_rem, random_state = 42)
    
    
    
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
