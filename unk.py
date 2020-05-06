#importing basic libraries
import re
import math 
import numpy as np

xxx = []
DATA_DIR = "D:/reserch/new models/bbc/business"
keyword = 'house'
import os
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        print(os.path.join(DATA_DIR+"/", file))
        xx = os.path.join(DATA_DIR+"/", file)
        xxx.append(xx)
'''
#defining method to calculate angle between document and query
def calc_angle(x,y):
    norm_x = np.linalg.norm(x)  # document vector magnitude calculation
    norm_y = np.linalg.norm(y)  # query vector magnitude calculation
    cos_theta = np.dot(x,y)/(norm_x*norm_y)  # calculating cosine of the angle between query and document
    theta = math.degrees(math.acos(cos_theta))  #inverse cosine to get the angle
    return theta
'''
#making the dictionary of the document
def word_dictionary(document):
    dictionary = {}
    with open(document, 'r') as file:
        
        text = file.read().lower()
        # replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
        text = re.sub('[^a-z\ \']+', " ", text)  # For some reason, even though the text is in lower case, the code does't work unless i redo that condition
        Words = list(text.split())  # put text into an empty list using split()
        for i in Words:
            if i in dictionary:
                dictionary[i] += 1
            else:
                dictionary[i] = 1
    return dictionary

# making the inverted index of the document
def make_invertedIndex(document):
    inverted = {}
    with open(document, 'r') as f:
        lines = f.read().splitlines()   # making a list of all documents seperated by a newline character

    idx = 1                                         # maintaining the current document index
    for docs in lines:
        doc_words = list(docs.split())               
        # for each word in documents
        for word in doc_words:
            if word in inverted:                    # if the word exists in the inverted index
                if idx not in inverted[word]:       # if current document is not in the value of this word
                    inverted[word].append(idx)      # add the current document as a value for the current word
            else:
                inverted[word] = [idx]              # if word is not a key of invertedindex then make a new key
        idx += 1;
    return inverted


for iii in xxx:
    with open(iii , 'r') as f:
        lines = f.read().splitlines()

# making the dictionary
    dictionary = word_dictionary(iii)
    print(iii)
    print ("Words in dictionary:  " , len(dictionary))
    
    # making the inverted index
    inverted = make_invertedIndex(iii)
    
    # comparing with queries
for root, dirs, files in os.walk(DATA_DIR, onerror=None): 
    for filename in files:  
        file_path = os.path.join(root, filename)
        #dictionary = make_dictionary(iii)
        #print ("Words in dictionary:  " , len(dictionary))
        try:
            with open(file_path, "rb") as f:  # open the file for reading
                
                for line in f:  
                    try:
                        line = line.decode("utf-8")
                        #lines = f.read().splitlines()
                        
                    except ValueError:  # decoding failed, skip the line
                        continue
                    if keyword in line:  # if the keyword exists on the current line...
                        print('=========================================')
                        print('keyword  :',keyword,' : contains in')
                        print('contents:')
                        #print(len(dictionary))
                        print(line)
                        print(file_path)  
                        continue  
        except (IOError, OSError):  
            pass
'''
try:
    
    with open('queries.txt', 'r') as f:
        queries = f.read().splitlines()         # getting individual queries from the quer.txt seperated by a newline

# for each query do the following
    for query in queries:

        print ("Query:  ", query)
        print ("Relevant documents: " ,)
    


        query_words = list(query.split())   
        print(query_words)    # split the query into individual words
        doc_all_query = inverted[query_words]# get all documents containing the first query word from inverted index
        print(doc_all_query)
        print('--------------------')
        for idx,query_word in enumerate(query_words):   # go through all the remaining words in the query
            doc_this_query = inverted[query_words[idx]] # get all documents of the word
            doc_all_query = [doc for doc in doc_all_query if doc in doc_this_query] # remove the documents that are not present in the next words

    # now we have only those documents which contain all the words from the given query
            print (' '.join(map(str, doc_all_query)))    # print all those documents

    # measuringthe similarity between each document we have got from above aginst the given query
            angleDict = {}
            queryDict = dict.fromkeys(dictionary, 0)
            for i in query_words:
                if i in queryDict:
                    queryDict[i] = 1
                    print(queryDict[i])
                    print('--------------------')

            queryVec = np.fromiter(queryDict.values(),dtype=float)  # changing this dictionary in a vector of only values as we dont need the keys now
            print(queryVec)
            # for each remaining document calculate this similarity
            for docs in doc_all_query:
                docDict = dict.fromkeys(dictionary, 0)
                Words = list(lines[docs-1].split())
                for i in Words:
                    if i in docDict:
                        docDict[i] += 1
        
                docVec = np.fromiter(docDict.values(), dtype=float) # calculating the document vector now
                angleDict[docs] = calc_angle(queryVec, docVec)  # passing the vectors in the function to get the similarity angle
        
            # sorting angles in ascending order and then printing
            angleSorted = sorted(angleDict, key=angleDict.get, reverse=False)
            for r in angleSorted:
                print (r, '{:.2f}'.format(round(angleDict[r], 2)))
except  :
    print()
'''  