#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lichunhui
@Time:  2018/4/10 16:09
@Description: N-gram SVM,NBayes,MaxEnt
"""

import sys
import multiprocessing

import nltk
import yaml
import numpy as np
import jieba
import keras
import pandas as pd
import re
from src import log
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml

np.random.seed(1337)  # For Reproducibility
sys.setrecursionlimit(1000000)

W2V_MODEL_PATH = 'lstm_data/w2v_model_{}.pkl'
LSTM_YML_PATH = 'lstm_data/lstm_{}.yml'
LSTM_MODEL_PATH = 'lstm_data/lstm_{}.h5'


class LSTMClassifier:
    vocab_dim = 100  # 词向量维度
    maxlen = 140  # 文本保留最大长度
    n_iterations = 1  # 词向量训练次数，可以多训练几次
    n_exposures = 10  # 词向量训练过程中，词频小于10的词语将会被忽略
    window_size = 7  # 词向量训练过程中窗口的大小，主要是决定某个单词被窗口内的词有关
    batch_size = 32  # 指定进行梯度下降时每个batch包含的样本数
    n_epoch = 10  # 训练的轮数，每个epoch会把训练集轮一遍
    input_length = 140
    cpu_count = multiprocessing.cpu_count() or 4

    # 去除类似soooo 保留soo
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)

    # twitter中@***这种将被替换成__HAND_
    hndl_regex = re.compile(r"@(\w+)")

    # twitter中话题#***这种将被替换成__HASH_
    hash_regex = re.compile(r"#(\w+)")

    # twitter中话题#***这种将被替换成__URL_
    url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

    # twitter中存在的一些表情
    emoticons = [('__EMOT_SMILE_', [':-)', ':)', '(:', '(-:', '^_^', '>y<', '>o<', '>O<', '^.^']),
                 ('__EMOT_LAUGH_', [':-D', ':D', 'X-D', 'XD', 'xD', ':p', ':P']),
                 ('__EMOT_LOVE_', ['<3', ':\*', '<33', '<333']),
                 ('__EMOT_WINK_', [';-)', ';)', ';-D', ';D', '(;', '(-;', '←_←', '→_→', '<_<', '>_>']),
                 ('__EMOT_SAD_', [':-(', ':(', '=_=', 'D:<', '</3', ':<']),
                 ('__EMOT_CRY_', [':,(', ':\'(', ':"(', ':((', 't_t', 'T_T', '→_←', 'T^T', '>_<']),
                 ]

    @classmethod
    def url_repl(cls, match):
        return '__URL_'

    @classmethod
    def hash_repl(cls, match):
        return '__HASH_' + match.group(1).upper()

    @classmethod
    def hndl_repl(cls, match):
        return '__HAND_'

    @classmethod
    def rpt_repl(cls, match):
        return match.group(1) + match.group(1)

    @classmethod
    def regex_union(cls, arr):
        return '(' + '|'.join(arr) + ')'

    @classmethod
    def escape_paren(cls, arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    negtn_regex = re.compile(r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|
                wont|wouldnt|dont|doesnt|didnt|isnt|aret|aint)$)|n't""", re.X)  # 否定检查

    @classmethod
    def get_negation_features(cls, words):
        INF = 0.0
        negtn = [bool(cls.negtn_regex.search(w)) for w in words]

        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0, len(words)):
            if (negtn[i]):
                prev = 1.0
            left[i] = prev
            prev = max(0.0, prev - 0.1)

        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0, len(words))):
            if (negtn[i]):
                prev = 1.0
            right[i] = prev
            prev = max(0.0, prev - 0.1)

        return dict(zip(['neg_l(' + w + ')' for w in words] + ['neg_r(' + w + ')' for w in words], left + right))

    @classmethod
    # 使用正则分词（没有做停用词处理）
    def text_parse(cls, x):
        try:
            sentence = x.strip().lower()
        except:
            sentence = x

        sentence = re.sub(cls.hndl_regex, cls.hndl_repl, sentence)  # 匹配替换@***
        sentence = re.sub(cls.hash_regex, cls.hash_repl, sentence)  # 匹配替换#***
        sentence = re.sub(cls.url_regex, cls.url_repl, sentence)  # 匹配替换URL
        sentence = re.sub(cls.rpt_regex, cls.rpt_repl, sentence)  # 匹配替换类似yoooooooo为yoo

        emoticons_regex = [(repl, re.compile(cls.regex_union(cls.escape_paren(regx)))) for (repl, regx) in
                           cls.emoticons]  # 匹配替换表情
        for (repl, regx) in emoticons_regex:
            sentence = re.sub(regx, ' ' + repl + ' ', sentence)

        pattern = r""" (?x)(?:[a-z]\.)+ 
                           | \d+(?:\.\d+)?%?\w+
                           | \w+(?:[-']\w+)*
                           | (?:[-.!?]{2,})
                           | [][.,;"'?():$-_*`]"""
        word_list = nltk.regexp_tokenize(sentence, pattern)
        return word_list

    # 加载训练文件
    @classmethod
    def load_file(cls):
        neg = pd.read_excel('corpus/negT.xlsx', header=None, index=None)
        pos = pd.read_excel('corpus/posT.xlsx', header=None, index=None)

        # pos[1] 即excle表格中第二列
        cw = lambda x: cls.text_parse(x)  # 定义分词函数
        pos['words'] = pos[1].apply(cw)
        neg['words'] = neg[1].apply(cw)
        # print(pos)
        print("=====> load files! pos length =", len(pos), "neg length =", len(neg))
        combined = np.concatenate((pos['words'], neg['words']))
        # 1表示正面情感，0表示负面情感
        y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))  # 合并语料
        return combined, y

    # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
    @classmethod
    def create_dictionaries(cls, model=None, combined=None):
        """ Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries
        """

        def _parse_dataset(sentences):
            """Words become integers"""
            data = []
            for sentence in sentences:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except KeyError:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        if combined is not None and model is not None:
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
            w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量
            combined = _parse_dataset(combined)
            combined = sequence.pad_sequences(combined, maxlen=cls.maxlen)  # 每个句子所含词语对应的索引，所有句子中含有频数小于10的词语，索引为0
            return w2indx, w2vec, combined
        else:
            print('No data provided...')

    # 训练词向量，创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
    @classmethod
    def word2vec_train(cls, combined, use_word_dim):
        model = Word2Vec(size=cls.vocab_dim, min_count=cls.n_exposures, window=cls.window_size,
                         workers=cls.cpu_count, iter=cls.n_iterations)
        model.build_vocab(combined)
        model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
        model.save(W2V_MODEL_PATH.format(use_word_dim))  # 词向量保存
        index_dict, word_vectors, combined = cls.create_dictionaries(model=model,
                                                                     combined=combined)  # 索引字典：{单词: 索引数字} ；词向量：{单词: 词向量(100维长的数组)}
        return index_dict, word_vectors, combined

    @classmethod
    def get_data(cls, index_dict, word_vectors, combined, y):
        n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, cls.vocab_dim))  # 索引为0的词语，词向量全为0
        for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = word_vectors[word]
        x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
        return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

    @classmethod
    def train_lstm(cls, n_symbols, embedding_weights, x_train, y_train, x_test, y_test, use_word_dim):
        model = Sequential()
        model.add(Embedding(output_dim=cls.vocab_dim, input_dim=n_symbols, mask_zero=True,
                            weights=[embedding_weights], input_length=cls.input_length))  # Adding Input Length
        model.add(LSTM(units=50, activation='sigmoid', recurrent_activation='hard_sigmoid'))
        model.add(Dropout(0.5))  # 防止过拟合的
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型

        model.fit(x_train, y_train, batch_size=cls.batch_size, epochs=cls.n_epoch,
                  verbose=1, validation_data=(x_test, y_test))  # 训练模型
        score = model.evaluate(x_test, y_test, batch_size=cls.batch_size)  # 评估模型

        yaml_string = model.to_yaml()
        with open(LSTM_YML_PATH.format(use_word_dim), 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights(LSTM_MODEL_PATH.format(use_word_dim))
        print('Test score:', score)

    # 训练模型
    @classmethod
    def train(cls):
        combined, y = cls.load_file()
        index_dict, word_vectors, combined = cls.word2vec_train(combined, str(cls.vocab_dim))  # 训练不同维度“vocab_dim”的词向量
        n_symbols, embedding_weights, x_train, y_train, x_test, y_test = cls.get_data(index_dict, word_vectors,
                                                                                      combined, y)
        cls.train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, str(cls.vocab_dim))

    @classmethod
    def input_transform(cls, string, usage):
        words = jieba.lcut(string)
        words = np.array(words).reshape(1, -1)
        model = Word2Vec.load(W2V_MODEL_PATH.format(usage))
        _, _, combined = cls.create_dictionaries(model, words)
        return combined

    @classmethod
    def predict(cls, sentence, use_word_dim):
        with open(LSTM_YML_PATH.format(use_word_dim), 'r') as f:
            yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        model.load_weights(LSTM_MODEL_PATH.format(use_word_dim))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        data = cls.input_transform(sentence, use_word_dim)
        data.reshape(1, -1)
        # print data
        result = model.predict_classes(data)
        print(model.predict(data))
        return result[0][0]


if __name__ == '__main__':
    LSTMClassifier.train()
