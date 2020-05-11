# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:06:00 2020

@author: HP
"""

import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
#import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

random_state = 0

DATA_DIR = "D:/reserch/fortraining/latest"
fname = "D:/reserch/fortraining/latest/Fundamental Rights/9.txt"

data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace", random_state=random_state)
df = pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])
print(df)
#test['tweet'].apply(lambda x: [item for item in x if item not in stop])     


def only_nouns(texts):
    output = []
    for doc in nlp.pipe(texts):
        noun_text = " ".join(token.lemma_ for token in doc if token.pos_ == 'NOUN' and 'VERB')
        output.append(noun_text)
        #print(noun_text)
    return output
    print(output)


df['text'] = only_nouns(df['text'])

df.head(5000)
n_topics = 12

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vec = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.5, min_df=1)
features = vec.fit_transform(df.text)

from sklearn.decomposition import NMF
cls = NMF(n_components=n_topics, random_state=random_state)
cls.fit(features)
# list of unique words found by the vectorizer
feature_names = vec.get_feature_names()
#print(feature_names)

# number of most influencing words to display per topic
n_top_words = 14

for i, topic_vec in enumerate(cls.components_):
    print(i, end=' ')
   
    for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:
        print(feature_names[fid], end=' ')
    print()
'''  
aa = []
with open(fname) as f:
    for line in f:
        aa.read(line)
    #print(content_array)
'''
f = open(fname, 'r')
document = f.read()
#stemmer = WordNetLemmatizer()


# Remove all the special characters
document = re.sub(r'\W', ' ', document)

# remove all single characters
document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

# Remove single characters from the start
document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

# Substituting multiple spaces with single space
document = re.sub(r'\s+', ' ', document, flags=re.I)

# Removing prefixed 'b'
document = re.sub(r'^b\s+', '', document)

# Converting to Lowercase
document = document.lower()
print(document)
    
new_articles = [
    document
    
]
#print(file_contents)
# and take the last one (with the highest score) for each row using `[:,-1]` indexing
cls.transform(vec.transform(new_articles)).argsort(axis=1)[:,-1]
print(cls.transform(vec.transform(new_articles)).argsort(axis=1)[:,-1])


'''
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(df)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(df[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
print(documents)
#Term frequency = (Number of Occurrences of a word)/(Total words in the document)

#IDF(word) = Log((Total number of documents)/(Number of documents containing the word))

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
#classifier.fit(X_train, y_train) 

'''