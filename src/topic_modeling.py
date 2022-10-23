

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:17:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""
# python3 -m spacy download enimport numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
# Gensim
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
import pprint
stop_words = stopwords.words('english')

num_topics = [7,10,15,20,25,30]
num_keywords = 15
train_df = pd.read_feather('../data/processed/full_clean.feather')


# We should try not to do too much of text pre-processing, because most of the questions are short, removing more words risks of losing meaning.


def prep_data_for_topic_modeling(df):
    def clean_text_for_topic(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text
    
    df.drop('index', axis=1, inplace=True) # As we read from feather, there is extra column index
    df['q1_topic_cleaned'] = pd.DataFrame(df.question1.apply(lambda x: clean_text_for_topic(x)))
    df['q2_topic_cleaned'] = pd.DataFrame(df.question2.apply(lambda x: clean_text_for_topic(x)))
    data = df['q1_cleaned'].values.tolist() + df['q2_cleaned'].values.tolist() # Concatenate q1 and q2 into one column
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def bigrams(words, bi_min=15):
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

def get_topic(caption, topics):
    result=lda_model[train_id2word.doc2bow(caption)][0]
    d={}
    d[result[0]]=result[1]
    key=max(d, key=d.get)
    return topics[key]


def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 10);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);

def scan_topic_modeling(num_topics):
    # Searches for optimal topic count based on coherence
    LDA_models = {}
    LDA_topics = {}
    for i in num_topics:
        LDA_models[i] =  gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                 id2word=train_id2word,
                                 num_topics=i,
                                 update_every=1,
                                 chunksize=len(train_corpus),
                                 passes=20,
                                 alpha='auto',
                                 random_state=20)

        shown_topics = LDA_models[i].show_topics(num_topics=i, 
                                                 num_words=num_keywords,
                                                 formatted=False)
        LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
    coherences = [CoherenceModel(model=LDA_models[i], texts=bigram_train, dictionary=train_id2word, coherence='c_v').get_coherence()\
                  for i in num_topics[:-1]]
    return coherences

def build_topic_model(num_topics, train_corpus, train_id2word, verbose = False):
    # Build gensim model
    lda_model = gensim.models.ldamodel.LdaModel(
                              num_topics = num_topics, # Number of topics        
                              corpus = train_corpus,
                              id2word = train_id2word, 
                              random_state=20,      
                              passes = 30, #how many times the algorithm is supposed to pass over the whole corpus
                              alpha = 'auto', # to let it learn the priors
                              update_every=1, # update the model every update_every chunksize chunks
                              chunksize = len(train_corpus), #number of documents to consider at once (affects the memory consumption)
                              )
    coherence_model_lda = CoherenceModel(model=lda_model, texts=bigram_train, dictionary=train_id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    if verbose: print('\nCoherence Score: ', coherence_lda)
    return lda_model

if __name__ ==  '__main__':
    # Search for optimal topic count
    coh_list = scan_topic_modeling(num_topics)
    data = prep_data_for_topic_modeling(train_df)
    train = pd.DataFrame({'text':data})
    train_corpus, train_id2word, bigram_train = get_corpus(train)

    lda_model = build_topic_model(25, train_corpus, train_id2word, True)
    doc_lda = lda_model[train_corpus]
    pd.set_option('display.max_columns', None)  
    pprint.pprint(get_lda_topics(lda_model, 25))

    topics={0:'Social Media/Gadget/Email',1:'Self-help/Learn/Business',2:'Purpose/Energy',3:'Language/Relationship',4:'Food/Health',5:'Interview/Difference/Drug',6:'Year/New/Stock/Company', 
        7:'Job/College/University', 8: 'India/Government/China', 9: 'English/Law/Writing', 10: 'Money/Bank/Online', 11: 'Relationship/Girl/Guy/People/Life',
        12: 'Politics/Trump/Election', 13: 'Assessment/Word/Home', 14: 'Country/Car/Show/Television', 15: 'Free/Ocatopm/Software/Website', 16: 'Engine/Password/Search',
        17: 'Long/Review/Work/Compare', 18: 'Best/Way/Visit', 19: 'Lose/Weight/Time/Travel/Salary', 20: 'Quora/Question/Google/Answer', 21: 'Problem/Increase', 22: 'Sex/Woman/Man',
        23: 'Movie/Video Game/Youtube', 24: 'United States/Day'}

    lemm_dfq1 = pd.DataFrame({'Text':bigram_train[0:404287]})
    lemm_dfq2 = pd.DataFrame({'Text':bigram_train[404287:]})

    train_df['q1_cleaned'] = lemm_dfq1['Text']
    train_df['q2_cleaned'] = lemm_dfq2['Text']

    train_df['q1_cleaned'].explode().dropna().groupby(level=0).agg(list)
    train_df['q2_cleaned'].explode().dropna().groupby(level=0).agg(list)

    train_df.loc[:, 'q1_topic']=train_df['q1_cleaned'].apply(lambda x: get_topic(x, topics) if type(x)==list else 'None')
    train_df.loc[:, 'q2_topic']=train_df['q2_cleaned'].apply(lambda x: get_topic(x, topics) if type(x)==list else 'None')


