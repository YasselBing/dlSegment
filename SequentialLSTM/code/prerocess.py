# -*- coding:utf-8 -*-
'''
generate train data for word segment
Created By Yassel Bing
2017-8-22
'''

import os
import codecs
from sklearn.externals import joblib


def load_data(input_file, output_file):
    '''
    为每个汉字添加标记
    :param input_file: 原始训练数据
    :param output_file: 保存路径
    :return: None
    '''
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/S ")
            else:
                output_data.write(word[0] + "/B ")
                for w in word[1:len(word) - 1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word) - 1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()


if __name__ == '__main__':
    input_file = '../data/test_msr_training_text'
    output_file = '../data/test_msr_training_taged'
    load_data(input_file, output_file)

