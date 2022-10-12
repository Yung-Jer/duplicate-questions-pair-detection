# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 23:17:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""

import pandas as pd

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

def build_lcs(df, verbose = False):
    if verbose: print(df['q1_cleaned'].apply(lambda x: len(x)).describe())
    if verbose: print(df['q2_cleaned'].apply(lambda x: len(x)).describe())
    
    if verbose: print("Building longest common substring and longest common subsequence...")
    df['q1_start'] = df['q1_cleaned'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else '')
    df['q2_start'] = ['q2_cleaned'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else '')
    
    df['lc_substring'] = df.apply(lambda x: longestCommonSubstring(x['q1_cleaned'], x['q2_cleaned']), axis=1)
    df['lc_subsequence'] = df.apply(lambda x: longestCommonSubsequence(x['q1_cleaned'], x['q2_cleaned']), axis=1)
    return df

if __name__ == '__main__':
    topic_df = pd.read_feather('../data/processed/train_w_topic_model.feather')

    # remove null values first
    topic_df.dropna(inplace=True)
    topic_df = build_lcs(topic_df)
    #topic_df.to_feather('../data/processed/train_w_topic_model_w_clean_data_w_lcs.feather')