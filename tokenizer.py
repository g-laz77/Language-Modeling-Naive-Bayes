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
    temp = bigrams[i][0] + '~' + bigrams[i][1]
    if bigrams_dict.get(temp) == None:
        bigrams_dict[temp] = 1
    else:
        bigrams_dict[temp] += 1
bg_vocab_len = len(bigrams_dict)

bigrams_probs = dict()
for i in range(len(bigrams)):
    temp = bigrams[i][0] + '~' + bigrams[i][1]
    if bigrams_probs.get(temp) == None:
        bigrams_probs[temp] = bigrams_dict[temp]/vocabulary[bigrams[i][0]]

for i in range(len(trigrams)):
    temp = trigrams[i][0] + '~' + trigrams[i][1] + '~' + trigrams[i][2]
    if trigrams_dict.get(temp) == None:
        trigrams_dict[temp] = 1
    else:
        trigrams_dict[temp] += 1
trg_vocab_len = len(trigrams_dict)

trigrams_probs =  dict()
for i in range(len(trigrams)):
    temp = trigrams[i][0] + '~' + trigrams[i][1] + '~' + trigrams[i][2]
    if trigrams_probs.get(temp) == None:
        trigrams_probs[temp] = trigrams_dict[temp]/bigrams_dict[trigrams[i][0]+'~'+trigrams[i][1]]

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
    plt.plot(x_axis,zipf_y, label="Unigram")         
    plt.legend(loc="upper right")
    plt.title("Zipf curves")
    plt.show()
    plt.plot(log_x_axis,log_zipf_y,label="Unigram")
    plt.legend(loc="upper right")
    plt.title("Zipf log-log curve")
    plt.show()

make_zipf()

def laplace_smoothing():                    #laplace smoothing        
    for i in range(len(list_tokens)):
        if ll_unigrams_probs.get(list_tokens[i]) == None:
            ll_unigrams_probs[list_tokens[i]] = (vocabulary[list_tokens[i]]+1)/(len(list_tokens)+vocab_length)

    for i in range(len(bigrams)):
        temp = bigrams[i][0] + '~' + bigrams[i][1]
        if ll_bigrams_probs.get(temp) == None:
            ll_bigrams_probs[temp] = (bigrams_dict[temp]+1)/(vocabulary[bigrams[i][0]]+vocab_length)

    for i in range(len(trigrams)):
        temp = trigrams[i][0] + '~' + trigrams[i][1] + '~' + trigrams[i][2]
        if ll_trigrams_probs.get(temp) == None:
            ll_trigrams_probs[temp] = (trigrams_dict[temp]+1)/(bigrams_dict[trigrams[i][1]+'~'+trigrams[i][2]]+ len(list_tokens)-1)


def smoothing_curve(uni_probs,bi_probs,tri_probs,Type):
    uni_sort_freq = sorted(uni_probs.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in uni_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y,label="Unigram")
    # plt.show()

    bi_sort_freq = sorted(bi_probs.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in bi_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y,label="Bigram")
    # plt.show()

    tri_sort_freq = sorted(tri_probs.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in tri_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y,label="Trigram")
    plt.legend(loc="upper right")
    plt.title("Zipf curves: "+Type)
    plt.show()

ll_unigrams_probs = dict()
ll_bigrams_probs = dict()
ll_trigrams_probs =  dict()
laplace_smoothing()
smoothing_curve(ll_unigrams_probs,ll_bigrams_probs,ll_trigrams_probs,"Laplace")

def wb_prob(word1,word2,word3,level):
    if level == 3:
        count = 0
        for key in trigrams_dict.keys():
            k_spl = key.split("~")
            if k_spl[2] == word3:
                count += 1
        onelambda = count / (count + len(list_tokens)-2)
        lamb = 1.0 - onelambda  
        if wb_bigrams_probs.get(word2+"~"+word3) == None:
           wb_bigrams_probs[word2+"~"+word3] = wb_prob(word2,word3,"",2)   
        return ((lamb*trigrams_probs[word1+"~"+word2+"~"+word3]) + (onelambda)*wb_bigrams_probs[word2+"~"+word3])
    elif level == 2:
        count = 0
        for key in bigrams_dict.keys():
            k_spl = key.split("~")
            if k_spl[1] == word2:
                count += 1
        onelambda = count / (count + len(list_tokens)-1)
        lamb = 1 - onelambda  
        # print(lamb,bigrams_probs[word1+"~"+word2])
        return ((lamb*bigrams_probs[word1+"~"+word2]) + (onelambda)*unigram_probs[word2])

def witten_bell():
    for i in range(len(trigrams)):
        temp = trigrams[i][0] + '~' + trigrams[i][1] +'~' + trigrams[i][2]
        if wb_trigrams_probs.get(temp) == None:
            wb_trigrams_probs[temp] = wb_prob(trigrams[i][0],trigrams[i][1],trigrams[i][2],3)
    
    for i in range(len(bigrams)):
        temp = bigrams[i][0] + '~' + bigrams[i][1]
        if wb_bigrams_probs.get(temp) == None:
            wb_bigrams_probs[temp] = wb_prob(bigrams[i][0],bigrams[i][1],"",2)
    
    for i in range(len(list_tokens)):
        if wb_unigrams_probs.get(list_tokens[i]) == None:
            wb_unigrams_probs[list_tokens[i]] = unigram_probs[list_tokens[i]]

wb_unigrams_probs = dict()
wb_bigrams_probs = dict()
wb_trigrams_probs =  dict()
witten_bell()
smoothing_curve(wb_unigrams_probs,wb_bigrams_probs,wb_trigrams_probs,"Witten Bell")

#kneser ney
def kn_prob(word1,word2,word3,level,discount=0.5):
    count = 0
    c2 = 0
    if level == 3:
        term1 = max((trigrams_dict[word1+"~"+word2+"~"+word3] - discount) / vocabulary[word1],0)
        for key in trigrams_dict.keys():
            k = key.split('~')
            if k[0] == word1 and k[1] == word2:
                count += 1
        
        term2 = discount * (count/bigrams_dict[word1+"~"+word2])
        count = count2 = 0
        for key in trigrams_dict.keys():
            k = key.split('~')
            # if k[1] == word2:
            #     print(k[1],word2)
            if k[1] == word2 and k[2] == word3:
                count += 1
            if k[1] == word2:
                count2 += 1
        
        term3 = max((count-discount),0)/count2
        count = count3 = 0
        for key in (bigrams_dict.keys()):
            k = key.split('~')
            if k[0] == word2:
                count += 1 
            if k[1] == word3:
                count3 += 1
        
        term4 = discount * (count/count2) * (count3/len(bigrams_dict)) 
        return (term1 + term2 * (term3+term4))

    elif level == 2:
        if (bigrams_dict.get(word1+"~"+word2) == None) or ((bigrams_dict[word1+"~"+word2] - discount) < 0):
            for key in bigrams_dict.keys():
                k_spl = key.split("~")
                if k_spl[1] == word2:
                    count += 1
                if k_spl[0] == word1:
                    c2 += 1
            prob_cont = count / len(bigrams_dict.keys())
            alpha = (discount/vocabulary[word1])*c2
            return (prob_cont * alpha)
        else:
            return ((bigrams_dict[word1+"~"+word2] - discount) / vocabulary[word1])

def kneser_Neys():
    for i in range(len(trigrams)):
        temp = trigrams[i][0] + '~' + trigrams[i][1] +'~' + trigrams[i][2]
        if kn_trigrams_probs.get(temp) == None:
            kn_trigrams_probs[temp] = kn_prob(trigrams[i][0],trigrams[i][1],trigrams[i][2],3)
    
    for i in range(len(bigrams)):
        temp = bigrams[i][0] + '~' + bigrams[i][1]
        if kn_bigrams_probs.get(temp) == None:
            kn_bigrams_probs[temp] = kn_prob(bigrams[i][0],bigrams[i][1],"",2)

    for i in range(len(list_tokens)):
        if kn_unigrams_probs.get(list_tokens[i]) == None:
            kn_unigrams_probs[list_tokens[i]] = max(vocabulary[list_tokens[i]] - 0.5,0)/len(list_tokens)

kn_unigrams_probs = dict()
kn_bigrams_probs = dict()
kn_trigrams_probs =  dict() 
kneser_Neys()
smoothing_curve(kn_unigrams_probs,kn_bigrams_probs,kn_trigrams_probs,"Kneser Neys")

def compare(uni1,uni2,uni3):
    uni_sort_freq = sorted(uni1.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in uni_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y,label="Witten Bell")
    uni_sort_freq = sorted(uni2.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in uni_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y,label="laplace")
    uni_sort_freq = sorted(uni3.items(), key = operator.itemgetter(1),reverse=True)
    curve_x = list()
    curve_y = list()
    i = 0
    for item in uni_sort_freq:
        curve_x.append(i)
        i += 1
        curve_y.append(item[1])
    plt.plot(curve_x,curve_y,label="Kneser Neys")
    plt.legend(loc="upper right")
    plt.title("Unigram comparison for all 3 Smoothing techniques")
    plt.show()

compare(wb_unigrams_probs,ll_unigrams_probs,kn_unigrams_probs)

#naive Bayes
file1_tokens = tokenize("entertainment_anime.txt")
file2_tokens = tokenize("lifestyle_food.txt")
file3_tokens = tokenize("news_conservative.txt")

file1_bi,file1_tri = language_modeling(file1_tokens)
file2_bi,file2_tri = language_modeling(file2_tokens)
file3_bi,file3_tri = language_modeling(file3_tokens)

file1_vocab = dict()
file2_vocab = dict()
file3_vocab = dict()
file1_bigram = dict()
file2_bigram = dict()
file3_bigram = dict()
file1_trigram = dict()
file2_trigram = dict()
file3_trigram = dict()

for i in range(len(file1_tokens)):
    if file1_vocab.get(file1_tokens[i]) == None:
        file1_vocab[file1_tokens[i]] = 1
    else:
        file1_vocab[file1_tokens[i]] += 1

for i in range(len(file2_tokens)):
    if file2_vocab.get(file2_tokens[i]) == None:
        file2_vocab[file2_tokens[i]] = 1
    else:
        file2_vocab[file2_tokens[i]] += 1

for i in range(len(file3_tokens)):
    if file3_vocab.get(file3_tokens[i]) == None:
        file3_vocab[file3_tokens[i]] = 1
    else:
        file3_vocab[file3_tokens[i]] += 1

for i in range(len(file1_bi)):
    if file1_bigram.get(file1_bi[i][0]+"~"+file1_bi[i][1]) == None:
        file1_bigram[file1_bi[i][0]+"~"+file1_bi[i][1]] = 1
    else:
        file1_bigram[file1_bi[i][0]+"~"+file1_bi[i][1]] += 1

for i in range(len(file2_bi)):
    if file2_bigram.get(file2_bi[i][0]+"~"+file2_bi[i][1]) == None:
        file2_bigram[file2_bi[i][0]+"~"+file2_bi[i][1]] = 1
    else:
        file2_bigram[file2_bi[i][0]+"~"+file2_bi[i][1]] += 1

for i in range(len(file3_bi)):
    if file3_bigram.get(file3_bi[i][0]+"~"+file3_bi[i][1]) == None:
        file3_bigram[file3_bi[i][0]+"~"+file3_bi[i][1]] = 1
    else:
        file3_bigram[file3_bi[i][0]+"~"+file3_bi[i][1]] += 1

for i in range(len(file1_tri)):
    if file1_trigram.get(file1_tri[i][0]+"~"+file1_tri[i][1]+"~"+file1_tri[i][2]) == None:
        file1_trigram[file1_tri[i][0]+"~"+file1_tri[i][1]+"~"+file1_tri[i][2]] = 1
    else:
        file1_trigram[file1_tri[i][0]+"~"+file1_tri[i][1]+"~"+file1_tri[i][2]] += 1

for i in range(len(file2_tri)):
    if file2_trigram.get(file2_tri[i][0]+"~"+file2_tri[i][1]+"~"+file2_tri[i][2]) == None:
        file2_trigram[file2_tri[i][0]+"~"+file2_tri[i][1]+"~"+file2_tri[i][2]] = 1
    else:
        file2_trigram[file2_tri[i][0]+"~"+file2_tri[i][1]+"~"+file2_tri[i][2]] += 1

for i in range(len(file3_tri)):
    if file3_trigram.get(file3_tri[i][0]+"~"+file3_tri[i][1]+"~"+file3_tri[i][2]) == None:
        file3_trigram[file3_tri[i][0]+"~"+file3_tri[i][1]+"~"+file3_tri[i][2]] = 1
    else:
        file3_trigram[file3_tri[i][0]+"~"+file3_tri[i][1]+"~"+file3_tri[i][2]] += 1

for key in file1_trigram.keys():
    k = key.split("~")
    file1_trigram[key] = file1_trigram[key] / file1_bigram[k[0]+"~"+k[1]]

for key in file2_trigram.keys():
    k = key.split("~")
    file2_trigram[key] = file2_trigram[key] / file2_bigram[k[0]+"~"+k[1]]

for key in file3_trigram.keys():
    k = key.split("~")
    file3_trigram[key] = file3_trigram[key] / file3_bigram[k[0]+"~"+k[1]]

for key in file1_bigram.keys():
    k = key.split("~")
    file1_bigram[key] = file1_bigram[key] / file1_vocab[k[0]]

for key in file2_bigram.keys():
    k = key.split("~")
    file2_bigram[key] = file2_bigram[key] / file2_vocab[k[0]]

for key in file3_bigram.keys():
    k = key.split("~")
    file3_bigram[key] = file3_bigram[key] / file3_vocab[k[0]]

# def make_zipf():                        #plotting the zipf curve
sort_freq = sorted(file1_vocab.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1]/len(file1_tokens))
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y,'r', label="File-1-Unigram")         

sort_freq = sorted(file2_vocab.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1]/len(file2_tokens))
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y, 'b',label="File-2-Unigram")

sort_freq = sorted(file3_vocab.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1]/len(file3_tokens))
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y, 'g',label="File-3-Unigram")
# plt.legend(loc="upper right")
# plt.show()

sort_freq = sorted(file1_bigram.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1])#/file1_vocab[temp[0]])
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
# print(zipf_yclea[0],x_axis[0])
plt.plot(x_axis,zipf_y, label="File-1-Bigram")         

sort_freq = sorted(file2_bigram.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1])
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y,label="File-2-Bigram")

sort_freq = sorted(file3_bigram.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1])
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y,label="File-3-Bigram")
# plt.legend(loc="upper right")
# plt.show()

sort_freq = sorted(file1_trigram.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1])
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y, label="File-1-Trigram")         

sort_freq = sorted(file2_trigram.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1])
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y,label="File-2-Trigram")

sort_freq = sorted(file3_trigram.items(), key = operator.itemgetter(1),reverse=True)
zipf_x = list()
zipf_y = list()
log_zipf_y = list()
for item in sort_freq:
    zipf_x.append(item[0])
    zipf_y.append(item[1])
    # log_zipf_y.append(math.log(item[1]))    

length = len(zipf_x)
x_axis = [i for i in range(len(zipf_x))]
log_x_axis = [math.log(i+0.5) for i in range(len(zipf_x))]
x_axis = np.array(x_axis)
plt.plot(x_axis,zipf_y,label="File-3-Trigram")
plt.legend(loc="upper right")
plt.title("Zipf curves for all 3 Text Sources")
plt.show()


#IOB model
class_dict = dict()
char_dict = dict()
classes = ['I','E','O','B','S']

#training
with open("out.csv") as f:
  while True:
    line = f.readline()
    line = line.split(',')
    if class_dict.get(line[1]) == None:
        class_dict[line[1]] = 1
    else:
        class_dict[1] += 1

    if char_dict.get(line[0]+"_"+line[1]) == None:
        char_dict[line[0]+"_"+line[1]] = 1
    else:
        char_dict[line[0]+"_"+line[1]] += 1

#testing
fil1 = open("lifestyle.txt")
text = fil1.readlines(10)




#testing




