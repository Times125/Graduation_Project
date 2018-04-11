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
        pos = pd.read_excel('corpus/posss.xlsx', header=None, index=None)
        neg = pd.read_excel('corpus/negss.xlsx', header=None, index=None)

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
    def bigramf(cls, words, score_fn=BigramAssocMeasures.chi_sq, n=2000):
        bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
        bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前n的双词
        new_bigrams = [u + ' ' + v for (u, v) in bigrams]

        return cls.unigramf(new_bigrams)

    @classmethod
    # 方案3：二元模型+单词；把二元和单词结合作为特征，并使用卡方统计的方法，选择排名前n的双词
    def unigram_and_bigramf(cls, words, score_fn=BigramAssocMeasures.chi_sq, n=2000):
        bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
        bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前n的双词
        new_bigrams = [u + ' ' + v for (u, v) in bigrams]
        a = cls.unigramf(words)
        b = cls.unigramf(new_bigrams)
        a.update(b)
        return a

    @classmethod
    # 方案4：一元模型+最优特征权重的特征选择（利用卡方统计）
    def unigram_chi(cls, pos, neg, n=2000):
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

        for word, freq in word_tf.items():
            pos_score = BigramAssocMeasures.chi_sq(con_word_tf['pos'][word], (freq, pos_word_count),
                                                   total_word_count)  # 计算积极词的卡方统计量,这里也可以计算互信息等其它统计量
            neg_score = BigramAssocMeasures.chi_sq(con_word_tf['neg'][word], (freq, neg_word_count),
                                                   total_word_count)  # 计算消极词的卡方统计量
            word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
        best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                    :n]  # 把词按信息量倒序排序。n是特征的维度，是可以不断调整直至最优的
        best_words = set([w for w, s in best_vals])
        print(best_words)
        return dict([(word, True) for word in best_words])

    @classmethod
    # 构建特征
    def build_features(cls, flag=0, number=2000):
        pos, neg = cls.load_file()  # 载入数据

        pos_words = list(itertools.chain(*pos))
        neg_words = list(itertools.chain(*neg))
        pos_words.extend(neg_words)
        dataset = pos_words
        print(len(pos), "===", len(neg), "===", len(dataset))
        pos_feature = []
        neg_feature = []
        if flag == 0:
            features = cls.unigramf(dataset)  # 单个词作为特征

            for items in pos:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = True
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = True
                negword = [a, 'neg']
                neg_feature.append(negword)

        elif flag == 1:
            features = cls.bigramf(dataset, n=number)  # 二元特征
            for items in pos:
                a = {}
                try:
                    bigram_finder = BigramCollocationFinder.from_words(items)  # 把文本变成双词搭配的形式
                    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)  # 使用卡方统计的方法，选择排名前n的双词
                except:
                    continue
                new_bigrams = [u + ' ' + v for (u, v) in bigrams]
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = True
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                try:
                    bigram_finder = BigramCollocationFinder.from_words(items)  # 把文本变成双词搭配的形式
                    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)  # 使用卡方统计的方法，选择排名前n的双词
                except:
                    continue
                new_bigrams = [u + ' ' + v for (u, v) in bigrams]
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = True
                negword = [a, 'neg']
                neg_feature.append(negword)
        elif flag == 2:
            features = cls.unigram_and_bigramf(dataset, n=number)  # 一元和二元特征
            for items in pos:
                a = {}
                bigram_finder = BigramCollocationFinder.from_words(items)  # 把文本变成双词搭配的形式
                bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)  # 使用卡方统计的方法，选择排名前n的双词
                new_bigrams = [u + ' ' + v for (u, v) in bigrams]
                new_bigrams = new_bigrams + items
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = True
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                bigram_finder = BigramCollocationFinder.from_words(items)  # 把文本变成双词搭配的形式
                bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)  # 使用卡方统计的方法，选择排名前n的双词
                new_bigrams = [u + ' ' + v for (u, v) in bigrams]
                new_bigrams = new_bigrams + items
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = True
                negword = [a, 'neg']
                neg_feature.append(negword)
        elif flag == 3:
            features = cls.unigram_chi(pos, neg, n=number)  # 权重+一元特征
            for items in pos:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = True
                posword = [a, 'pos']
                pos_feature.append(posword)

            for items in neg:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = True
                negword = [a, 'neg']
                neg_feature.append(negword)

        return pos_feature, neg_feature

    @classmethod
    # 计算分类器的准确度
    def score(cls, mclassifier, x_train, x_test):
        data, tag = zip(*x_test)  # 分离测试集合的数据和标签，便于验证和测试
        classifier = SklearnClassifier(mclassifier)
        classifier.train(x_train)
        print('Trian done!')
        pred = classifier.classify_many(data)  # 给出预测的标签
        n = 0
        s = len(pred)
        for i in range(0, s):
            if pred[i] == tag[i]:
                n = n + 1
        return n / s  # 分类器准确度
        pass

    @classmethod
    def record_res(cls, filename, taskname, f, n):
        pos_feature, neg_feature = Ngram.build_features(flag=f, number=n)
        shuffle(pos_feature)
        shuffle(neg_feature)
        index = int((len(pos_feature) + len(neg_feature)) * 0.2)
        x_train = pos_feature[index:] + neg_feature[index:]
        x_test = pos_feature[:index] + neg_feature[:index]

        a = Ngram.score(BernoulliNB(), x_train, x_test)
        b = Ngram.score(MultinomialNB(), x_train, x_test)
        c = Ngram.score(LogisticRegression(), x_train, x_test)
        d = Ngram.score(SVC(), x_train, x_test)
        e = Ngram.score(LinearSVC(), x_train, x_test)
        f = Ngram.score(NuSVC(), x_train, x_test)
        log.console_out(filename, taskname, n, a, b, c, d, e, f)
        print('all done!')


if __name__ == '__main__':
    tasks = ['unigram', 'bigram_best', 'uni_bigram', 'unigram_best']
    ns = [2500, 5000, 7500, 10000]
    for i in range(4):
        filename = tasks[i] + '.txt'
        if tasks[i] == tasks[0]:
            Ngram.record_res(filename, tasks[i], i, 0)
        elif tasks[i] == tasks[1]:
            for n in ns:
                Ngram.record_res(filename, tasks[i], i, n)
        elif tasks[i] == tasks[2]:
            for n in ns:
                Ngram.record_res(filename, tasks[i], i, n)
        elif tasks[i] == tasks[3]:
            for n in ns:
                Ngram.record_res(filename, tasks[i], i, n)
    # print('BernoulliNB`s accuracy is %f' % Ngram.score(BernoulliNB(), x_train, x_test))
    # print('MultinomiaNB`s accuracy is %f' % Ngram.score(MultinomialNB(), x_train, x_test))
    # print('LogisticRegression`s accuracy is  %f' % Ngram.score(LogisticRegression(), x_train, x_test))
    # print('SVC`s accuracy is %f' % Ngram.score(SVC(), x_train, x_test))
    # print('LinearSVC`s accuracy is %f' % Ngram.score(LinearSVC(), x_train, x_test))
    # print('NuSVC`s accuracy is %f' % Ngram.score(NuSVC(), x_train, x_test))
