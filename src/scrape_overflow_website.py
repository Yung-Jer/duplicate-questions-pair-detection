from bs4 import BeautifulSoup
import time
import pandas as pd
import requests
import os

def find_questions(url, page_size):
    """Find questions based on tag and sort by criteria."""
    print('Starting crawling...')
    links = []
    question = []
    related_question = []

    df = pd.DataFrame()
    
    for page_no in range(1, page_size+1):
        time.sleep(2)
        source_code = requests.get(f'{url}{page_size}').text
        soup = BeautifulSoup(source_code, 'html.parser')
        for question_link in soup.find_all('a', {'class': 'question-hyperlink'}):
            links.append(question_link['href'])
    
    for link in links:
        time.sleep(2)
        source_code = requests.get(link).text
        soup = BeautifulSoup(source_code, 'html.parser')
        
        #Find Question Title
        q1 = soup.find('a', {'class': 'question-hyperlink'})
        question.append(q1.get_text())
        

        #Find Related Question
        related = soup.find("div", {"class": "related js-gps-related-questions"})
        question2 = related.get_text().splitlines()
        question2[:] = [x for x in question2 if x]
        related_question.append(question2[1])
        
    df = pd.DataFrame(list(zip(links, question, related_question)))
    df.columns =['Links', 'Question1', 'Related_Question']
    return df


questions = find_questions(f'http://stackoverflow.com/questions?tab=frequent&page=', 1)



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
questions.to_csv("../data/raw/scraped_data.csv", index = False)