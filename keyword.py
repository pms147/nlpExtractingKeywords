import math
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
import sys

# file = open("C:\\Users\\Sang\\source\\repos\\Lab01B\\bbc-fulltext\\bbc\\tech\\002.txt", encoding="utf8")
# data = file.read()
# file.close()

def lower_case(data):
    res = data.casefold()
    return res


def removing_num(data):
    res = re.sub(r'\d', '', data)
    return res


def removing_punctuations(data):
    res = re.sub(r'[^\w\s]', '', data)
    return res


def removing_whitespace(data):
    res = re.sub(r'^\s+|\s+$', '', data)
    return res


def tokenization(data):
    word_tokens = word_tokenize(data)
    return word_tokens


# print(data)
# print(word_tokens)


def removing_stopwords(word_tokens):
    stop_words = set(stopwords.words('english'))
    res = []
    for word in word_tokens:
        if word not in stop_words:
            res.append(word)
    return res

# lemmatization
def lemmatization(b):
    lemmatizer = WordNetLemmatizer()
    c=[]
    for w in b:
        c.append(lemmatizer.lemmatize(w))
    return c
# stemming
def stemming(c):
    ps = PorterStemmer()

    d=[]
    for w in c:
        d.append(ps.stem(w))
    return d



def calculate_tf(word, words_in_document):
    tfDict = {}
    wordsCount = len(words_in_document)
    for word, count in word.items():
        tfDict[word] = math.log10(1+count/float(wordsCount))
    return tfDict

def calculate_idf(corpus):
    idfDict = {}
    N = len(corpus)
    
    idfDict = dict.fromkeys(corpus[0].keys(), 0)
    for doc in corpus:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict

def computeTFIDF(corpus, top_key):
    tfidf = {}
    for word, val in top_key.items():
        tfidf[word] = val*corpus[word]
    return tfidf



def process(filename,data,top_k):
    data = lower_case(data)
    data = removing_num(data)
    data = removing_punctuations(data)
    data = removing_whitespace(data)
    word_tokens = tokenization(data)
    b = removing_stopwords(word_tokens)
    d = lemmatization(b)
    

    wordDictA = dict.fromkeys(d, 0)
    wordDictB = dict.fromkeys(d, 0)
    
    for word in d:
        wordDictA[word]+=1
    tf_word = calculate_tf(wordDictA, d)
    idfs = calculate_idf([wordDictA,wordDictB])
    tfidf = computeTFIDF(tf_word, idfs)
    
    data=sorted(tfidf.items(), key = lambda x : x[1],reverse=True)[:top_k]
    df = pd.DataFrame(data, columns = ['Word', 'TF-IDF'])
    
    temp = [(filename),(str(df['Word'].values))]
    print(temp)
    result = pd.DataFrame([temp], columns=['filename', 'keywords'])
    result.transpose() 
    return result

for i in range(1,402):
    path = sys.argv[1]
    if (i < 10):
        filename = f'00{i}.txt'
    elif (i < 100):
        filename = f'0{i}.txt'  
    else:  
        filename = f'{i}.txt'  
    
    data = open(path+filename, encoding="utf8").read()
    output = process(filename,data,5)
    output.to_csv(sys.argv[2], mode='a', header=False, index=False)

# python keyword.py bbc-fulltext/bbc/tech/ output.csv