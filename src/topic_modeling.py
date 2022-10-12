

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
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
stop_words = stopwords.words('english')


TOTAL_TOPICS = 20
train_df = pd.read_feather('../data/processed/train_clean.feather')


# We should try not to do too much of text pre-processing, because most of the questions are short, removing more words risks of losing meaning.


def prep_data_for_topic_modeling(df):
    df.drop('index', axis=1, inplace=True)
    df['q1_topic_cleaned'] = pd.DataFrame(df.question1.apply(lambda x: clean_text_for_topic(x)))
    df['q2_topic_cleaned'] = pd.DataFrame(df.question2.apply(lambda x: clean_text_for_topic(x)))
    data = df['q1_cleaned'].values.tolist() + df['q2_cleaned'].values.tolist()
    return data
    
    def clean_text_for_topic(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text

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

def get_topic(caption, lda_model, topics):
    result=lda_model[train_id2word.doc2bow(caption)][0]
    d={}
    for i in result:
        d[i[0]]=i[1]
    key=max(d, key=d.get)
    return topics[key]

data = prep_data_for_topic_modeling(train_df)
train = pd.DataFrame({'text':data})
train_corpus, train_id2word, bigram_train = get_corpus(train)


"""
lda_model = gensim.models.ldamodel.LdaModel(
                          num_topics = 15, # Number of topics        
                          corpus = train_corpus,
                          id2word = train_id2word, 
                          random_state=20,      
                          passes = 10, #how many times the algorithm is supposed to pass over the whole corpus
                          alpha = 'auto', # to let it learn the priors
                          update_every=1, # update the model every update_every chunksize chunks
                          chunksize = 100, #number of documents to consider at once (affects the memory consumption)
                          per_word_topics=True,
                          )
pprint(lda_model.print_topics())
doc_lda = lda_model[train_corpus]

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=bigram_train, dictionary=train_id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


topics={0:'People',1:'Work/Tech/App',2:'Life',3:'Politics',4:'Knowledge',5:'Education',6:'Language/Good', 
        7:'India/Food', 8: 'One/Feeling', 9: 'Job/Design', 10: 'World/Country/War', 11: 'Year/Age/Experience',
        12: 'Money/Sex', 13: 'Movie/Ever/Imrpove', 14: 'Time/Travel'}


lemm_dfq1 = pd.DataFrame({'Text':bigram_train[0:404287]})
lemm_dfq2 = pd.DataFrame({'Text':bigram_train[404287:]})

train_df['q1_cleaned'] = lemm_dfq1['Text']
train_df['q2_cleaned'] = lemm_dfq2['Text']

train_df['q1_cleaned'].explode().dropna().groupby(level=0).agg(list)
train_df['q2_cleaned'].explode().dropna().groupby(level=0).agg(list)


train_df.loc[:, 'q1_topic']=train_df['q1_cleaned'].apply(lambda x: get_topic(x, lda_model, topics) if type(x)==list else 'None')
train_df.loc[:, 'q2_topic']=train_df['q2_cleaned'].apply(lambda x: get_topic(x, lda_model, topics) if type(x)==list else 'None')

train_df.reset_index().to_feather('../data/processed/train_w_topic_model.feather')
"""
# Try this 
#https://stackoverflow.com/questions/32313062/what-is-the-best-way-to-obtain-the-optimal-number-of-topics-for-a-lda-model-usin

# Scan for optimal parameters
def calculate_coherence_score(n, alpha, beta):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus,
                                           id2word=train_id2word,
                                           num_topics=n, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha=alpha,
                                           per_word_topics=True,
                                           eta = beta)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=bigram_train, dictionary=train_id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda

#list containing various hyperparameters
no_of_topics = [12,15,20]
alpha_list = ['symmetric',0.3,0.5,0.7]
beta_list = [0.3,0.5,0.7]

# n : 7 ; alpha : symmetric ; beta : 0.5 ; Score : 0.2914646846171962 highest so far
for n in no_of_topics:
    for alpha in alpha_list:
        for beta in beta_list:
            coherence_score = calculate_coherence_score(n, alpha, beta)
            print(f"n : {n} ; alpha : {alpha} ; beta : {beta} ; Score : {coherence_score}")


"""
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


"""

