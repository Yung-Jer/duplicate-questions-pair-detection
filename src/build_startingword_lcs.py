import pandas as pd

import os
  
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

topic_df = pd.read_feather('../data/processed/train_w_topic_model.feather')

# remove null values first
topic_df.dropna(inplace=True)
print(topic_df['q1_cleaned'].apply(lambda x: len(x)).describe())
print(topic_df['q2_cleaned'].apply(lambda x: len(x)).describe())

topic_df['q1_start'] = topic_df['q1_cleaned'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else '')
topic_df['q2_start'] = topic_df['q2_cleaned'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else '')

def longestCommonSubsequence(text1: str, text2: str) -> int:
        _seen = {}
        n, m = len(text1), len(text2)
        
        def dp(i, j):
            if i == n or j == m:
                return 0
            elif (i, j) in _seen:
                return _seen[(i, j)]
            if text1[i] == text2[j]:
                res = 1 + dp(i+1, j+1)
            else:
                res = max(dp(i+1, j), dp(i, j+1))
            _seen[(i, j)] = res
            return res
        
        return dp(0, 0)

def longestCommonSubstring(text1: str, text2: str) -> int:
        _seen = {}
        n, m = len(text1), len(text2)
        
        def dp(i, j):
            if i == n or j == m:
                return 0
            elif (i, j) in _seen:
                return _seen[(i, j)]
            count = 0
            if text1[i] == text2[j]:
                count = 1 + dp(i+1, j+1)
            count = max(count, dp(i+1, j), dp(i, j+1))
            _seen[(i, j)] = count
            return count
        
        return dp(0, 0)

topic_df['lc_substring'] = topic_df.apply(lambda x: longestCommonSubstring(x['q1_cleaned'], x['q2_cleaned']), axis=1)
topic_df['lc_subsequence'] = topic_df.apply(lambda x: longestCommonSubsequence(x['q1_cleaned'], x['q2_cleaned']), axis=1)

# topic_df.to_feather('../data/processed/train_w_topic_model_w_clean_data_w_lcs.feather')