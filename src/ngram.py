#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lichunhui
@Time:  2018/4/10 16:09
@Description: N-gram SVM,NBayes,MaxEnt
"""
import re
import time
import nltk
import sklearn
import itertools
import pandas as pd
from src import log
from random import shuffle
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import TrigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Ngram:
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
                 ('__EMOT_LAUGH_', [':-D', ':D', 'X-D', 'XD', 'xD', ':p', ':P' ]),
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
    # 一元模型，单个词作为特征
    def unigramf(cls, words):
        return dict([(word, 1) for word in words])

    @classmethod
    # 二元模型;把双个词作为特征
    def bigramf(cls, words):
        bigrams = nltk.bigrams(words)
        new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
        return cls.unigramf(new_bigrams)

    @classmethod
    # 三元模型;把三个词作为特征
    def trigramf(cls, words):
        trigrams = nltk.trigrams(words)
        new_trigrams = ['has(%s,%s,%s)' % (w, u, v) for (w, u, v) in trigrams]
        return cls.unigramf(new_trigrams)

    @classmethod
    # 一元模型 + 二元模型
    def unigram_and_bigramf(cls, words):  # , score_fn=BigramAssocMeasures.chi_sq, n=2000):
        # bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
        # bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前n的双词
        a = cls.unigramf(words)
        b = cls.bigramf(words)
        a.update(b)
        return a

    @classmethod
    # 一元模型 + 二元模型 + 三元模型
    def unigram_and_bigram_trigramf(cls, words):

        a = cls.unigramf(words)
        b = cls.bigramf(words)
        c = cls.trigramf(words)
        a.update(b)
        a.update(c)
        return a

    @classmethod
    # 一元模型+最优特征权重的特征选择（利用卡方统计）
    def unigram_chi(cls, pos, neg, n=1000):
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
            word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量(即算出信息增益)
        best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[
                    :n]  # 把词按信息量倒序排序。n是特征的维度，是可以不断调整直至最优的
        best_words = set([w for w, s in best_vals])
        # print(best_words)
        return dict([(word, 1) for word in best_words])

    @classmethod
    # 构建特征
    def build_features(cls, flag=0, number=1000):

        pos, neg = cls.load_file()  # 载入数据
        pos_words = list(itertools.chain(*pos))
        neg_words = list(itertools.chain(*neg))
        pos_words.extend(neg_words)
        dataset = pos_words

        print(len(pos), "===", len(neg), "===", len(dataset))

        pos_feature = []
        neg_feature = []
        # 方案一：uni
        if flag == 0:
            features = cls.unigramf(dataset)
            for items in pos:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = 1
                # negation_features = cls.get_negation_features(items)  # 引入否定检查
                # a.update(negation_features)
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = 1
                # negation_features = cls.get_negation_features(items)
                # a.update(negation_features)
                negword = [a, 'neg']
                neg_feature.append(negword)
        # 方案二：bi
        elif flag == 1:
            features = cls.bigramf(dataset)
            for items in pos:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = 1
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = 1
                negword = [a, 'neg']
                neg_feature.append(negword)
        # 方案三：uni+bi
        elif flag == 2:
            features = cls.unigram_and_bigramf(dataset)  # 一元和二元特征
            for items in pos:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = 1
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = 1
                negword = [a, 'neg']
                neg_feature.append(negword)
        # 方案四：uni_info
        elif flag == 3:  # 最优一元模型
            features = cls.unigram_chi(pos, neg, n=number)  # 权重+一元特征
            for items in pos:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = 1
                posword = [a, 'pos']
                pos_feature.append(posword)

            for items in neg:
                a = {}
                for item in items:
                    if item in features.keys():
                        a[item] = 1
                negword = [a, 'neg']
                neg_feature.append(negword)
        # 方案五：uni_info + bi
        elif flag == 4:  # 最优一元模型 + 二元模型
            features = cls.unigram_chi(pos, neg, n=number)  # 最优一元模型
            uni_feats = cls.bigramf(dataset)  # 二元模型
            features.update(uni_feats)
            for items in pos:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = 1
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                for item in new_bigrams:
                    if item in features.keys():
                        a[item] = 1
                negword = [a, 'neg']
                neg_feature.append(negword)
        # 方案六：uni + bi + tri
        elif flag == 5:  # 一元模型 + 二元模型 + 三元模型
            features = cls.unigram_and_bigram_trigramf(dataset)
            for items in pos:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                trigrams = nltk.trigrams(items)
                new_trigrams = ['has(%s,%s,%s)' % (w, u, v) for (w, u, v) in trigrams]
                new_trigrams.extend(new_bigrams)
                for item in new_trigrams:
                    if item in features.keys():
                        a[item] = 1
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                trigrams = nltk.trigrams(items)
                new_trigrams = ['has(%s,%s,%s)' % (w, u, v) for (w, u, v) in trigrams]
                new_trigrams.extend(new_bigrams)
                for item in new_trigrams:
                    if item in features.keys():
                        a[item] = 1
                negword = [a, 'neg']
                neg_feature.append(negword)
        # 方案七：uni_info + bi + tri
        elif flag == 6:  # 最优一元模型 + 二元模型 + 三元模型
            features = cls.unigram_chi(pos, neg, n=number)  # 最优一元模型
            uni_feats = cls.bigramf(dataset)  # 二元模型
            tri_feats = cls.trigramf(dataset)  # 三元模型
            features.update(uni_feats)
            features.update(tri_feats)
            for items in pos:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                trigrams = nltk.trigrams(items)
                new_trigrams = ['has(%s,%s,%s)' % (w, u, v) for (w, u, v) in trigrams]
                new_trigrams.extend(new_bigrams)
                for item in new_trigrams:
                    if item in features.keys():
                        a[item] = 1
                posword = [a, 'pos']
                pos_feature.append(posword)
            for items in neg:
                a = {}
                bigrams = nltk.bigrams(items)
                new_bigrams = ['has(%s,%s)' % (u, v) for (u, v) in bigrams]
                new_bigrams.extend(items)
                trigrams = nltk.trigrams(items)
                new_trigrams = ['has(%s,%s,%s)' % (w, u, v) for (w, u, v) in trigrams]
                new_trigrams.extend(new_bigrams)
                for item in new_trigrams:
                    if item in features.keys():
                        a[item] = 1
                negword = [a, 'neg']
                neg_feature.append(negword)

        return pos_feature, neg_feature

    @classmethod
    # 计算分类器的准确度
    def score(cls, mclassifier, x_train, x_test):
        data, tag = zip(*x_test)  # 分离测试集合的数据和标签，便于验证和测试
        classifier = SklearnClassifier(mclassifier)
        classifier.train(x_train)
        print('===================================》》 Train done!')
        pos_index = []
        neg_index = []
        for i in range(0, len(tag)):
            if tag[i] == 'pos':  # pos
                pos_index.append(i)  # 记录所有pos的index，计算精确率
            else:
                neg_index.append(i)

        pred = classifier.classify_many(data)  # 给出预测的标签
        print(type(pred), len(pred), len(tag))
        n = 0
        s = len(pred)
        for i in range(0, s):
            if pred[i] == tag[i]:
                n = n + 1
        accu = n / s  # 分类器准确率
        print(accu)
        print(pos_index, tag)
        tp = 0  # 将正类预测为正类的数目
        fn = 0  # 将正类预测为负类的数目
        fp = 0  # 将负类预测为正类的数目
        tn = 0  # 将负类预测为负类的数目

        for i in pos_index:
            if pred[i] == tag[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        for i in neg_index:
            if pred[i] == tag[i]:
                tn = tn + 1
            else:
                fp = fp + 1
        print(tp, '--', fn, '--', fp, '--', tn)
        recall = tp / (tp + fn)  # 召回率

        f1 = (2 * tp) / (len(tag) + tp - tn)  # f1值
        return accu, recall, f1

    @classmethod
    def record_res(cls, filename, taskname, f, n):
        pos_feature, neg_feature = Ngram.build_features(flag=f, number=n)
        shuffle(pos_feature)
        shuffle(neg_feature)
        index = int(((len(pos_feature) + len(neg_feature)) / 2) * 0.2)
        x_train = pos_feature[index:] + neg_feature[index:]
        x_test = pos_feature[:index] + neg_feature[:index]

        a, aa, aaa = Ngram.score(BernoulliNB(), x_train, x_test)
        b, bb, bbb = Ngram.score(MultinomialNB(), x_train, x_test)
        c, cc, ccc = Ngram.score(LogisticRegression(), x_train, x_test)
        d, dd, ddd = Ngram.score(SVC(), x_train, x_test)
        e, ee, eee = Ngram.score(LinearSVC(), x_train, x_test)
        f, ff, fff = Ngram.score(NuSVC(), x_train, x_test)
        # 记录 准确率（Accuracy）、召回率（recall）、F1值
        log.console_out(filename, taskname, n, ('BNB', a, aa, aaa), ('MNB', b, bb, bbb), ('LR', c, cc, ccc),
                        ('SVC-rbf', d, dd, ddd), ('LSVC', e, ee, eee), ('NuSVC', f, ff, fff))


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %A %H:%M:%S',time.localtime(time.time())))
    # 7种搭配方案
    tasks = ['unigram', 'bigram', 'uni_bigram', 'unigram_info', 'uni_info_bigram', 'uni_bi_tri', 'uni_info_bi_tri']
    ns = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for i in range(0, 7):
        filename = 'record.txt'
        if tasks[i] == tasks[0]:  # uni
            Ngram.record_res(filename, tasks[i], i, 0)
        elif tasks[i] == tasks[1]:  # bi
            Ngram.record_res(filename, tasks[i], i, 0)
        elif tasks[i] == tasks[2]:  # uni_bi
            Ngram.record_res(filename, tasks[i], i, 0)
        elif tasks[i] == tasks[3]:  # uni_info
            for n in ns:
                Ngram.record_res(filename, tasks[i], i, n)
        elif tasks[i] == tasks[4]:  # uni_info_bi
            for n in ns:
                Ngram.record_res(filename, tasks[i], i, n)
        elif tasks[i] == tasks[5]:  # uni_bi_tri
            Ngram.record_res(filename, tasks[i], i, 0)
        elif tasks[i] == tasks[6]:  # uni_info_bi_tri
            for n in ns:
                Ngram.record_res(filename, tasks[i], i, n)
    log.console_out(filename, "All Tasks Done")
    print(time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime(time.time())))
