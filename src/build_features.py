# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:17:27 2022

@author: Calven Ng, Tay Xun Yang, Wong Yung Jer, Cheang Xue Ting, Tiara Lau
"""
import pandas as pd
import topic_modeling as tm
import build_startingword_lcs as sw
import time


# Variables Setting 
verbose = True
TOTAL_TOPICS = 20





if __name__ == '__main__':
    tic = time.time()
    # Read data
    train_df = pd.read_feather('../data/processed/train_clean.feather')
    
    # Build Longest Common Substring and Sub Sequence as a new feature
    lcs_df = sw.build_lcs(train_df, verbose)
    
    
    
    
    
    
    
    
    
    
    
    # End
    toc = time.time()
    time_taken = toc - tic
    print("Time Taken: " + time.strftime('%H:%M:%S', time.gmtime(time_taken)))
