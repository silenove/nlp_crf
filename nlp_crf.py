#coding=utf-8

import re
import jieba.posseg as pseg
import codecs
import numpy as np
from sklearn.model_selection import KFold
import os
import sys

file_read = codecs.open('data_nr.txt','r', 'utf-8')

raw_data = file_read.readlines()
sentences = np.array(raw_data)
file_read.close()

#构建训练数据
def create_train_data(words, fwrite):
    for word in words:
        word = word.strip(' ')
        if len(word) > 1:
            if '/nr' in word:
                word = word.rstrip('/nr')
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + 'nr' + '\t' + 'B' + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + 'nr' + '\t' + 'I' + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + 'nr' + '\t' + 'I' + '\n')
            else:
                wf = pseg.cut(word)
                w, flag = next(wf)
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + flag + '\t' + 'O' + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + flag + '\t' + 'O' + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + flag + '\t' + 'O' + '\n')
        elif len(word) == 1:
            wf = pseg.cut(word)
            w, flag = next(wf)
            fwrite.write(word + '\t' + 'S' + '\t' + flag + '\t' + 'O' + '\n')

#构建测试数据
def create_test_data(words, fwrite):
    for word in words:
        word = word.strip(' ')
        if len(word) > 1:
            if '/nr' in word:
                word = word.rstrip('/nr')
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + 'nr' + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + 'nr' + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + 'nr' + '\n')
            else:
                wf = pseg.cut(word)
                w, flag = next(wf)
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + flag + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + flag + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + flag + '\n')
        elif len(word) == 1:
            wf = pseg.cut(word)
            w, flag = next(wf)
            fwrite.write(word + '\t' + 'S' + '\t' + flag + '\n')

#判断预测输出和期望输出的相似度（预测正确的人名数 / 人名总数）
def evaluate_model(file_predict, file_expect):
    predict = codecs.open(file_predict, 'r', 'utf-8')
    expect = codecs.open(file_expect, 'r', 'utf-8')

    #人名总数
    total = 0.
    #识别正确数
    prec = 0.

    for line_exp in expect:
        line_pre = predict.readline()

        symbol_exp = line_exp.strip('\n').split('\t')[-1]
        if symbol_exp == 'O':
            continue
        else:
            total += 1

        if line_pre != '':
            symbol_pre = line_pre.strip('\n').split('\t')[-1]
            if symbol_exp == symbol_pre:
                prec += 1

    predict.close()
    expect.close()
    return prec / total

#交叉验证
print('train sentences: ', int(len(sentences)*0.8))
print('test sentences: ' , int(len(sentences)*0.2))

kf = KFold(n_splits=5)
epoch = 1
for train_index, test_index in kf.split(sentences):
    print('epoch : ', epoch)

    #输出文件
    file_train_write = codecs.open('train.txt', 'w', 'utf-8')
    file_test_write = codecs.open('test.txt', 'w', 'utf-8')
    #输出期望结果，用于评判模型性能
    file_expect = codecs.open('expect.txt', 'w', 'utf-8')

    raw_train_data = sentences[train_index]
    raw_test_data = sentences[test_index]

    #构建训练数据
    for line in raw_train_data:
        words = line.strip('\n').split(' ')
        create_train_data(words, file_train_write)
        file_train_write.write('\n')

    print('train data created')

    #构建测试数据和期望输出
    for line in raw_test_data:
        words = line.strip('\n').split(' ')
        create_test_data(words, file_test_write)
        create_train_data(words, file_expect)
        file_test_write.write('\n')
        file_expect.write('\n')

    print('test data created')

    file_train_write.close()
    file_test_write.close()
    file_expect.close()

    os.system('crf_learn -f 4 -c 1.5 temp.txt train.txt crf_model')
    os.system('crf_test -m crf_model test.txt > predict.txt')

    acc = evaluate_model('expect.txt', 'predict.txt')
    print('epoch :',epoch,',','test acc :',acc)
    
    epoch += 1

