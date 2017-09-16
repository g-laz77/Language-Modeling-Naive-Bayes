import re
import sys
import matplotlib.pyplot as plt
import operator
import numpy as np
import math
unigrams_dict = dict()
bigrams_dict = dict()
trigrams_dict = dict()
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
    unigrams_dict = vocabulary

    bigrams_dict = dict()
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

    return bigrams,trigrams

list_tokens = tokenize(file1)               #making the vocabulary
vocabulary = dict()
bigrams,trigrams = language_modeling(list_tokens)

for i in range(len(list_tokens)):
    if vocabulary.get(list_tokens[i]) == None:
        vocabulary[list_tokens[i]] = 1
    else:
        vocabulary[list_tokens[i]] += 1
vocab_length = len(vocabulary)

unigram_probs = dict()
for i in range(len(list_tokens)):
    if unigram_probs.get(list_tokens[i]) == None:
        unigram_probs[list_tokens[i]] = vocabulary[list_tokens[i]]/len(list_tokens)

for i in range(len(bigrams)):
    temp = bigrams[i][0] + '_' + bigrams[i][1]
    if bigrams_dict.get(temp) == None:
        bigrams_dict[temp] = 1
    else:
        bigrams_dict[temp] += 1
bg_vocab_len = len(bigrams_dict)

bigrams_probs = dict()
for i in range(len(bigrams)):
    temp = bigrams[i][0] + '_' + bigrams[i][1]
    if bigrams_probs.get(temp) == None:
        bigrams_probs[temp] = bigrams_dict[temp]/vocabulary[bigrams[i][0]]

for i in range(len(trigrams)):
    temp = trigrams[i][0] + '_' + trigrams[i][1] + '_' + trigrams[i][2]
    if trigrams_dict.get(temp) == None:
        trigrams_dict[temp] = 1
    else:
        trigrams_dict[temp] += 1
trg_vocab_len = len(trigrams_dict)

trigrams_probs =  dict()
for i in range(len(trigrams)):
    temp = trigrams[i][0] + '_' + trigrams[i][1] + '_' + trigrams[i][2]
    if trigrams_probs.get(temp) == None:
        trigrams_probs[temp] = trigrams_dict[temp]/bigrams_dict[trigrams[i][1]+'_'+trigrams[i][2]]

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

#make_zipf()

def laplace_smoothing():                    #laplace smoothing        
    for i in range(len(list_tokens)):
        if ll_unigrams_probs.get(list_tokens[i]) == None:
            ll_unigrams_probs[list_tokens[i]] = (vocabulary[list_tokens[i]]+1)/(len(list_tokens)+vocab_length)

    for i in range(len(bigrams)):
        temp = bigrams[i][0] + '_' + bigrams[i][1]
        if ll_bigrams_probs.get(temp) == None:
            ll_bigrams_probs[temp] = (bigrams_dict[temp]+1)/(vocabulary[bigrams[i][0]]+vocab_length)

    for i in range(len(trigrams)):
        temp = trigrams[i][0] + '_' + trigrams[i][1] + '_' + trigrams[i][2]
        if ll_trigrams_probs.get(temp) == None:
            ll_trigrams_probs[temp] = (trigrams_dict[temp]+1)/(bigrams_dict[trigrams[i][1]+'_'+trigrams[i][2]]+ len(list_tokens)-1)


def smoothing_curve(uni_probs,bi_probs,tri_probs):
    uni_sort_freq = sorted(uni_probs.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in uni_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y)
    plt.show()

    bi_sort_freq = sorted(bi_probs.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in bi_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y)
    plt.show()

    tri_sort_freq = sorted(tri_probs.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in tri_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y)
    plt.show()

# ll_unigrams_probs = dict()
# ll_bigrams_probs = dict()
# ll_trigrams_probs =  dict()
# laplace_smoothing()
# smoothing_curve(ll_unigrams_probs,ll_bigrams_probs,ll_trigrams_probs)

def wb_prob(word1,word2,word3,level):
    if level == 3:
        count = 0
        for key in trigrams_dict.keys():
            k_spl = key.split("_")
            if k_spl[2] == word3:
                count += 1
        onelambda = count / (count + len(list_tokens)-2)
        lamb = 1 - onelambda  
        if wb_bigrams_probs.get(word1+"_"+word2) == None:
           wb_bigrams_probs[word1+"_"+word2] = wb_prob(word1,word2,"",2)   
        return ((lamb*trigrams_probs[word1+"_"+word2+"_"+word3]) + (onelambda)*wb_bigrams_probs[word1+"_"+word2])
    elif level == 2:
        count = 0
        for key in bigrams_dict.keys():
            k_spl = key.split("_")
            if k_spl[1] == word2:
                count += 1
        onelambda = count / (count + len(list_tokens)-1)
        lamb = 1 - onelambda  
        return ((lamb*bigrams_probs[word1+"_"+word2]) + (onelambda)*unigram_probs[word1])


def witten_bell():
    for i in range(len(trigrams)):
        temp = trigrams[i][0] + '_' + trigrams[i][1] +'_' + trigrams[i][2]
        if wb_trigrams_probs.get(temp) == None:
            wb_trigrams_probs[temp] = wb_prob(trigrams[i][0],trigrams[i][1],trigrams[i][2],3)
    
    for i in range(len(bigrams)):
        temp = bigrams[i][0] + '_' + bigrams[i][1]
        if wb_bigrams_probs.get(temp) == None:
            wb_bigrams_probs[temp] = wb_prob(bigrams[i][0],bigrams[i][1],"",2)
    
    for i in range(len(list_tokens)):
        if wb_unigrams_probs.get(list_tokens[i]) == None:
            wb_unigrams_probs[list_tokens[i]] = unigram_probs[list_tokens[i]]

wb_unigrams_probs = dict()
wb_bigrams_probs = dict()
wb_trigrams_probs =  dict()
witten_bell()
smoothing_curve(wb_unigrams_probs,wb_bigrams_probs,wb_trigrams_probs)

def kn_prob(word1,word2,word3,level,discount = 0.75):
    count = 0
    c2 = 0
    if level == 2:
        if (bigrams_dict.get(word1+"_"+word2) == None) or ((bigrams_dict[word1+"_"+word2] - discount) < 0):
            for key in bigrams_dict.keys():
                k_spl = key.split("_")
                if k_spl[1] == word2:
                    count += 1
                if k_spl[0] == word1:
                    c2 += 1
            prob_cont = count / len(bigrams_dict.keys())
            alpha = (discount/vocabulary[word1])*c2
            kn_bigrams_probs = prob_cont * alpha
        else:
            kn_bigrams_probs = ((bigrams_dict[word1+"_"+word2] - discount) / vocabulary[word1])

def kneyser_Neys():
    for i in range(len(trigrams)):
        temp = trigrams[i][0] + '_' + trigrams[i][1] +'_' + trigrams[i][2]
        if kn_trigrams_probs.get(temp) == None:
            kn_trigrams_probs[temp] = kn_prob(trigrams[i][0],trigrams[i][1],trigrams[i][2],3)
    
    for i in range(len(bigrams)):
        temp = bigrams[i][0] + '_' + bigrams[i][1]
        if kn_bigrams_probs.get(temp) == None:
            kn_bigrams_probs[temp] = kn_prob(bigrams[i][0],bigrams[i][1],"",2)
    

kn_unigrams_probs = dict()
kn_bigrams_probs = dict()
# kn_trigrams_probs =  dict()    
