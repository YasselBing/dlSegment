# dlSegment
中文分词程序（Chinese Word Segment）
使用深度学习框架keras的序贯模型解决中文分词问题。

SequentialLSTM：

一共三个文件夹：
code：
preprocess：预处理，用于给训练数据标记；

data：
test_msr_training_text：训练数据（部分样例）
test_msr_training_taged：生成的带标记的数据

model：
msr_training_single_word.w2v.bin：单字的word2vector

包依赖：
numpy、gensim、keras

代码共分三个部分：
（1）准备数据
单字的vector、已分好词的训练数据
（2）数据预处理
给每个字加上BMES的标记
（3）训练数据生成
padding补充数据，防止后面的数据丢失、读取词向量、生成训练数据

网络结构：
model = Sequential()
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(4))
model.add(Activation("softmax"))


训练结果：
预测结果样例：

Reference：https://github.com/gcarq/keras-timeseries-prediction


