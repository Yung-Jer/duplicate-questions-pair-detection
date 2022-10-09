

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:17:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""
# python3 -m spacy download enimport numpy as np
import pandas as pd

import os
from nltk.corpus import stopwords
import re, nltk, spacy, string

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint
import numpy as np


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

TOTAL_TOPICS = 40
train_df_raw = pd.read_csv('../data/raw/train.csv')
train_df = train_df_raw.dropna()


# We should try not to do too much of text pre-processing, because most of the questions are short, removing more words risks of losing meaning.

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df = train_df

df['q1_cleaned'] = pd.DataFrame(df.question1.apply(lambda x: clean_text(x)))
df['q2_cleaned'] = pd.DataFrame(df.question2.apply(lambda x: clean_text(x)))


nlp = spacy.load('en_core_web_sm')
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)
    
df["q1_cleaned"] =  df.apply(lambda x: lemmatizer(x['q1_cleaned']), axis=1)
df["q2_cleaned"] =  df.apply(lambda x: lemmatizer(x['q2_cleaned']), axis=1)


data = df['q1_cleaned'].values.tolist() + df['q2_cleaned'].values.tolist()

# SKlearn
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=5, # minimum number of occurences            
                             stop_words='english',             
                             lowercase=True,                   
                             token_pattern='[a-zA-Z0-9]{3,}',  
                             max_features=5000, # max number of unique words
                            )

data_vectorized = vectorizer.fit_transform(
    data)

lda_model = LatentDirichletAllocation(n_components=15, # Number of topics
                                      learning_method='online',
                                      random_state=0,       
                                      max_iter=10,      
                                      batch_size=128, 
                                      evaluate_every=1,
                                      n_jobs = -1  # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=5):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=5)

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords

topics={0:'Win/Lose',1:'Life/long',2:'Start/Compare',3:'Time/Create',4:'Love/Say',5:'Way/Learn',
        6:'Indian/Girl/Buy/Account', 7:'Preparaton', 8: 'Work/Job/Company', 9: 'Day/Old',
        10: 'English/Person', 11: 'Phone/App', 12: 'Movie/Counrty/World', 13: 'Politics/People/Change', 14: 'Study/High'}


df_topic_keywords['topic_theme'] = topics
df_topic_keywords.set_index('topic_theme', inplace=True)
df_topic_keywords.T








# Gensim
import gensim
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
stop_words = stopwords.words('english')
stop_words.extend(['come','order','try','go','get','make','drink','plate','dish','restaurant','place',
                  'would','really','like','great','service','came','got'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def get_corpus(df):
    df['text'] = [re.sub('\s+', ' ', sent) for sent in df['text']]
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram_mod = bigrams(df.text)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


train = pd.DataFrame({'text':data})
train_corpus, train_id2word, bigram_train = get_corpus(train)



lda_model = gensim.models.ldamodel.LdaModel(
                          num_topics = 15, # Number of topics        
                          corpus = train_corpus,
                          id2word = train_id2word, 
                          random_state=20,      
                          passes = 50,
                          alpha = 'auto',
                          update_every=1,       
                          per_word_topics=True,
                          )
pprint(lda_model.print_topics())
doc_lda = lda_model[train_corpus]


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=bigram_train, dictionary=train_id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)