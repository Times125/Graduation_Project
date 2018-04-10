#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lichunhui
@Time:  2018/4/10 16:09
@Description: N-gram SVM,NBayes,MaxEnt
"""
import re
import nltk
import sklearn
import itertools
import pandas as pd
from src import log
from random import shuffle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Ngram:

    @classmethod
    # 使用正则分词（没有做停用词处理）
    def text_parse(cls, x):
        try:
            sentence = x.strip().lower()
        except:
            sentence = x
        sentence = re.sub(r'@\s*[\w]+ | ?#[\w]+ | ?&[\w]+; | ?[^\x00-\xFF]+', '', sentence)
        # 可以匹配一些颜文字
        pattern = r""" (?x)(?:[a-z]\.)+ 
                              | \d+(?:\.\d+)?%?\w+
                              | \w+(?:[-']\w+)*
                              | <\d+
                              | (?:[><*$]+[._]*[<>*$]+)
                              | [:;][\w*\d*][-]*[)(]*
                              | (?:[-.!?]{2,})
                              | [][.,;"'?():$-_*`]"""
        word_list = nltk.regexp_tokenize(sentence, pattern)
        return word_list

    @classmethod
    # 导入文件
    def load_file(cls):
        pos = pd.read_excel('corpus/posT.xlsx', header=None, index=None)
        neg = pd.read_excel('corpus/negT.xlsx', header=None, index=None)

        cutword = lambda x: cls.text_parse(x)  # 分词函数
        pos['word'] = pos[1].apply(cutword)
        neg['word'] = neg[1].apply(cutword)
        pos = list(pos['word'])  # [[],[],[]]
        neg = list(neg['word'])

        return pos, neg

    @classmethod
    # 方案1：一元模型，单个词作为特征
    def unigramf(cls, words):
        return dict([(word, True) for word in words])

    @classmethod
    # 方案2：二元模型;把双个词作为特征，并使用卡方统计的方法，选择排名前的双词
    def bigramf(cls, words, score_fn=BigramAssocMeasures.chi_sq, n=10000):
        bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
        bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前n的双词
        new_bigrams = [u + ' ' + v for (u, v) in bigrams]
        return cls.unigramf(new_bigrams)

    @classmethod
    # 方案3：二元模型+单词；把二元和单词结合作为特征，并使用卡方统计的方法，选择排名前n的双词
    def unigram_and_bigramf(cls, words, score_fn=BigramAssocMeasures.chi_sq, n=10000):
        bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
        bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前n的双词
        new_bigrams = [u + ' ' + v for (u, v) in bigrams]
        a = cls.unigramf(words)
        b = cls.unigramf(new_bigrams)
        a.update(b)
        return a

    @classmethod
    # 方案4：二元模型+最优特征权重的特征选择（利用卡方统计）
    def feature_choose(cls, n=10000):
        pos, neg = cls.load_file()
        pos_words = list(itertools.chain(*pos))
        neg_words = list(itertools.chain(*neg))

        word_tf = FreqDist()  # 统计所有词频
        con_word_tf = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频

        for word in pos_words:
            word_tf[word] += 1
            con_word_tf['pos'][word] += 1
        for word in neg_words:
            word_tf[word] += 1
            con_word_tf['neg'][word] += 1

        pos_word_count = con_word_tf['pos'].N()  # 积极词的数量
        neg_word_count = con_word_tf['neg'].N()  # 消极词的数量
        total_word_count = pos_word_count + neg_word_count  # 总词
        word_scores = {}  # 包括了每个词和这个词的信息量

        for word, freq in word_tf.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(con_word_tf['pos'][word], (freq, pos_word_count),
                                                   total_word_count)  # 计算积极词的卡方统计量,这里也可以计算互信息等其它统计量
            neg_score = BigramAssocMeasures.chi_sq(con_word_tf['neg'][word], (freq, neg_word_count),
                                                   total_word_count)  # 计算消极词的卡方统计量
            word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
        best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                    :n]  # 把词按信息量倒序排序。n是特征的维度，是可以不断调整直至最优的
        best_words = set([w for w, s in best_vals])
        return dict([(word, True) for word in best_words])

    @classmethod
    # 计算分类器的准确度
    def score(cls, classifer, x_train):
        classifer = SklearnClassifier(classifer)
        classifer.train(x_train)
        pass


if __name__ == '__main__':
    Ngram.load_file()
