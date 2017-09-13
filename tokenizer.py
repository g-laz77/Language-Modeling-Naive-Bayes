import re
import sys
import matplotlib.pyplot as plt
import operator
import numpy as np
import math

file1 = sys.argv[1]
scanner=re.Scanner([
  (r"[0-9]+",       lambda scanner,token:("INTEGER", token)),
  (r"https*\ \ [a-z_0-9]+",       lambda scanner,token:("URL", token)),   # for the entertainment_anime text
  (r"[a-z_A-Z]+",      lambda scanner,token:("IDENTIFIER", token)),
  (r"[,.]+",        lambda scanner,token:("PUNCTUATION", token)),
  (r"\s+", None),                                                         # None == skip token.
])

def tokenize(filename):
    f = open(filename,"r+")
    results, remainder=scanner.scan(f.read())#"I need to write a program in NLTK that breaks a corpus into unigrams, bigrams, trigrams, fourgrams and fivegrams.")
    tokens = list()
    for result in results:
        tokens.append(result[1])
    # print(tokens)
    return tokens

def language_modeling(tokens):
    unigrams = list()                #unigram model
    unigrams = tokens

    bigrams = list()
    for i in range(len(tokens)-1):   #bigram models
        temp = list()
        temp.append(tokens[i])
        temp.append(tokens[i+1])
        bigrams.append(temp)
    # print(bigrams)

    trigrams = list()
    for i in range(len(tokens)-2):   #trigram models
        temp = list()
        temp.append(tokens[i])
        temp.append(tokens[i+1])
        temp.append(tokens[i+2])
        trigrams.append(temp)


list_tokens = tokenize(file1)               #making the vocabulary
vocabulary = dict()
for i in range(len(list_tokens)):
    if vocabulary.get(list_tokens[i]) == None:
        vocabulary[list_tokens[i]] = 1
    else:
        vocabulary[list_tokens[i]] += 1


def make_zipf():                        #plotting the zipf curve
    sort_freq = sorted(vocabulary.items(), key = operator.itemgetter(1),reverse=True)

    zipf_x = list()
    zipf_y = list()
    log_zipf_y = list()
    for item in sort_freq:
        zipf_x.append(item[0])
        zipf_y.append(item[1])
        log_zipf_y.append(math.log(item[1]))    

    length = len(zipf_x)
    x_axis = [i for i in range(len(zipf_x))]
    log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
    x_axis = np.array(x_axis)
    plt.plot(x_axis,zipf_y, label="Zipf curve")         
    #plt.xticks(x_axis,zipf_x)
    plt.show()
    plt.plot(log_x_axis,log_zipf_y,label="logcurve")
    plt.show()

#def laplace_smoothing():


make_zipf()


