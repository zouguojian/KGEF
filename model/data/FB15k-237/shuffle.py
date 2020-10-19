import os
import random
divide = [0.4, 0.6]

filename = 'train.txt'

filelenght = 0

fr =  open(filename, 'r')
for line in fr:
    line = line.replace("\n","")
    filelenght += 1

for div in divide:
    list1 = [i for i in range(0,filelenght)]

    random.shuffle(list1)
    resuffle = list1[0: int(filelenght*div)]
    resuffle.sort()

    shuffle_len = len(resuffle)
    index = 0
    new_start = 0

    fr =  open(filename, 'r')
    fw = open('train' + str(div) +'.txt', 'a')
    
    for line in fr:
        if new_start == resuffle[index]:
            lines = line.replace("\n","")
            lines = str(lines) + "\n"
            fw.write(lines)
            index += 1
        if index == len(resuffle):
            break
        new_start += 1
