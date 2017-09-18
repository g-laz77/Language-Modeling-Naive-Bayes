import os
import sys

#IOB model
class_dict = dict()
char_dict = dict()
classes = ['I','E','O','B','S']

#training
with open("out.csv") as f:
  while True:
    line = f.readline()
    if line:
        line = line.split(',')
        if class_dict.get(line[1][:-1]) == None:
            class_dict[line[1][:-1]] = 1
        else:
            class_dict[line[1][:-1]] += 1

        if char_dict.get(line[0]+"_"+line[1][:-1]) == None:
            char_dict[line[0]+"_"+line[1][:-1]] = 1
        else:
            char_dict[line[0]+"_"+line[1][:-1]] += 1
    else:
        break

#print([key for key in char_dict.keys()])
#testing
fil1 = open("entertainment_anime.txt")
text = fil1.readlines(10)
# print(text)
# print(char_dict['n_I'])
for line in text:
    i = 0
    while line[i] != "\n":
        mx = 0
        classified = 'X'
        for clas in classes:
            if char_dict.get(line[i]+"_"+clas) == None:
                continue
            else:
                if char_dict[line[i]+"_"+clas]/class_dict[clas] > mx:
                    mx = char_dict[line[i]+"_"+clas]/class_dict[clas]
                    classified = clas
        
        print(line[i],classified)
        i += 1
            