# encoding:utf-8
'''
深度学习序列处理
Created By Yassel Bing
Date:2017-8-22
'''
import os
import numpy as np
import codecs
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Activation, GRU, Bidirectional,Input


class WordSegment(object):
    def __init__(self):

        self.model_path = '../model/w2v-word-segment.model'  # 模型存储路径
        self.taged_data_path = '../data/test_msr_training_taged'  # 预处理后的数据
        self.single_word_vector_path = '../model/msr_training_single_word.w2v.bin'  # 单字的vector

        self.label_id_dict = {u'S': 0, u'B': 1, u'M': 2, u'E': 3}  # label数值化
        self.num_label_dict = {n: l for l, n in self.label_id_dict.items()}  # 数值化表示的label，用于后面的规则判断

        self.nb_classes = 4  # 类别个数
        self.word_step = 5  # 用于标注的上下文大小，分词不用考虑太多的上下文，所以设置了5
        self.padding_count = self.word_step // 2  # 需要添加的padding个数

        self.w2v_model = KeyedVectors.load_word2vec_format(self.single_word_vector_path,
                                                           binary=True,
                                                           unicode_errors='ignore')  # 读取预先做好的词向量

        self.vocabs = self.w2v_model.vocab.keys()  # 获得所有词向量的key，用于判断是否有词向量

        random_vector = self.w2v_model[list(self.w2v_model.vocab.keys())[0]]  # 任意词向量，为了获取长度
        self.pad_arr = np.zeros_like(random_vector)  # 用于填充没有向量的字

    def load_taged_data(self):
        '''
        处理预处理后的数据作为模型的输入
        第一部分，padding补充数据，防止后面的数据丢失
        第二部分，读取词向量
        第三部分，生成训练数据

        :yield:train_x, train_y
        '''
        x_train = []  # 最终的x_train
        y_train = []  # 最终的y_train
        taged = codecs.open(self.taged_data_path, 'r', 'utf-8')
        for line in taged.readlines():
            word_tags = line.strip('\n').split()
            if not word_tags:
                continue
            word_tags = ["PADDING/S"] * self.padding_count + word_tags + ["PADDING/S"] * self.padding_count  # TODO:为什么要加两个末尾标记（OK）
            # 必须加足够的padding，否则如果shingling太大的话，最后面的内容如果不够shingling最小值将会被丢弃
            for i in range(len(word_tags) + 1 - self.word_step):  # 去掉所有内容都是padding的行
                context = word_tags[i:i + self.word_step]  # shingling

                # 获取词向量，判断tags里面是否有数字和字母，数字和字母的vector都是空

                train_temp = []
                for j, word_tag in enumerate(context):
                    word, tag = word_tag.split('/')
                    try:
                        word_vector = self.w2v_model[word]
                    except:
                        word_vector = self.pad_arr

                    if word.isdigit() or word.isalpha():
                        word_vector = self.pad_arr
                    train_temp.append(word_vector)

                word, tag = word_tags[i + self.padding_count].split("/")
                if word.isdigit() or word.isalpha():
                    tag = u'S'
                if len(train_temp) == self.word_step:  # 丢弃掉长度不够上下文长度的shingling
                    x_train.append(train_temp)
                    y = self.label_id_dict[tag]
                    y_train.append(y)
        x = np.array(x_train)
        self.x_train = x
        self.y_train = to_categorical(y_train, self.nb_classes)

    def model_lstm(self, dim):
        inputs = Input(shape=(dim, 200))
        lstm = LSTM(512, dropout=0.5)(inputs)
        dense1 = Dense(128)(lstm)
        pred = Dense(4, activation='softmax')(dense1)
        model = Model(inputs=inputs, outputs=pred)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 validation_split=0.2,
                                 epochs=1,
                                 batch_size=64)
        print(history)

    def predict(self, sentence):
        '''
        预测函数
        第一部分，padding补充数据，防止后面的数据丢失
        第二部分，读取词向量
        第三部分，预测结果
        第四部分，使用规则
        :param sentence: 需要分词的句子
        :return:
        '''
        self.tag_tmp = None
        sentence = list(sentence)
        sentence = ["PADDING"] * self.padding_count + sentence + ["PADDING"] * self.padding_count

        for i in range(len(sentence) + 1 - self.word_step):
            context = sentence[i:i + self.word_step]
            word = sentence[i + self.padding_count]
            word_vector_list = []
            for j, w in enumerate(context):
                if w not in self.vocabs:
                    word_vector = self.pad_arr
                else:
                    word_vector = self.w2v_model[w]
                if w.isalpha() or w.isdigit():
                    word_vector = self.pad_arr
                word_vector_list.append(word_vector)
            test_x = np.array([word_vector_list])

            prob = self.model.predict(test_x)
            prob_sort_list = prob.argsort().tolist()[0]
            prob_sort_list.reverse()
            for prob_i in prob_sort_list:
                tag = self.num_label_dict[prob_i]
                if self.tag_tmp is None and tag in [u"E", u"M"]: continue  # 词开头不能是E或M
                if self.tag_tmp == u"B" and tag in [u"B", u"S"]: continue  # B后不能接B
                if self.tag_tmp == u"E" and tag in [u"E", u"M"]: continue  # E后不能接E或M
                if self.tag_tmp == u"M" and tag in [u"B", u"S"]: continue  # M后不能接B或S
                if self.tag_tmp == u"S" and tag in [u"E", u"M"]: continue  # S后不能接E或M
                # if self.tag_tmp == u"ENG" and tag in [u"E", u"M"]: continue
                # if self.tag_tmp == u"NUM" and tag in [u"E", u"M"]: continue
                break
            self.tag_tmp = tag
            print("%-2d" %i, word, tag, prob_sort_list, prob)

if __name__ == '__main__':
    chinese_word_seg = WordSegment()
    chinese_word_seg.load_taged_data()
    chinese_word_seg.model_lstm(5)
    chinese_word_seg.predict('风险投资的目的不在于对被投资企业股份的占有和控制，而是尽快回收流动资金以回馈投资者并进行新的投资。')
