import re
import nltk
import numpy as np
import pandas as pd
from src import log
from sklearn.svm import SVC
from sklearn.externals import joblib
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics

"""
SVM 分类器 word2vec
"""


class SVMClassifer:

    @classmethod
    def load_file(cls):
        neg = pd.read_excel('corpus/Sentiment0.xlsx', header=None, index=None)
        pos = pd.read_excel('corpus/Sentiment1.xlsx', header=None, index=None)

        # pos[1] 即excle表格中第二列
        cw = lambda x: cls.text_parse(x)  # 定义分词函数
        pos['words'] = pos[1].apply(cw)
        neg['words'] = neg[1].apply(cw)
        # print(pos)
        log.console_out("log_svm.txt", "pos length =", len(pos), "neg length =", len(neg))
        # use 1 for positive sentiment, 0 for negative
        y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))  # 合并语料
        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y,
                                                            test_size=0.2)  # 划分训练和测试集合8/2
        # print(x_train, '\n', y_train)
        np.save('svm_data/y_train.npy', y_train)
        np.save('svm_data/y_test.npy', y_test)
        log.console_out("log_svm.txt", "load_file done!")
        return x_train, x_test

    @classmethod
    # 使用正则分词
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
    # 对每个句子的所有词向量取均值
    def build_wordvector(cls, text, size, comment_w2v):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in text:
            try:
                vec += comment_w2v[word].reshape((1, size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    # 计算词向量
    @classmethod
    def save_train_vecs(cls, x_train, x_test):
        n_dim = 128
        # Initialize model and build vocab
        comment_w2v = Word2Vec(size=n_dim, min_count=5)
        comment_w2v.build_vocab(x_train)

        # Train the model over train_reviews (this may take several minutes)
        comment_w2v.train(x_train, total_examples=comment_w2v.corpus_count, epochs=comment_w2v.iter)
        train_vecs = np.concatenate([cls.build_wordvector(z, n_dim, comment_w2v) for z in x_train])
        # train_vecs = scale(train_vecs)

        np.save('svm_data/train_vecs.npy', train_vecs)
        log.console_out("log_svm.txt", "save train vecs done!")

        # Train word2vec on test tweets
        comment_w2v.train(x_test, total_examples=comment_w2v.corpus_count, epochs=comment_w2v.iter)
        comment_w2v.save('svm_data/w2v_model/w2v_model.pkl')
        # Build test tweet vectors then scale
        test_vecs = np.concatenate([cls.build_wordvector(z, n_dim, comment_w2v) for z in x_test])
        # test_vecs = scale(test_vecs)
        np.save('svm_data/test_vecs.npy', test_vecs)

        log.console_out("log_svm.txt", "save test vecs done!")
        log.console_out("log_svm.txt", "save_train_vecs function done!")

    @classmethod
    def get_data(cls):
        train_vecs = np.load('svm_data/train_vecs.npy')
        y_train = np.load('svm_data/y_train.npy')
        test_vecs = np.load('svm_data/test_vecs.npy')
        y_test = np.load('svm_data/y_test.npy')
        return train_vecs, y_train, test_vecs, y_test

    # 训练svm模型
    @classmethod
    def train(cls):
        x_train, x_test = cls.load_file()
        cls.save_train_vecs(x_train, x_test)  # w2v计算词向量
        train_vecs, y_train, test_vecs, y_test = cls.get_data()
        clf = SVC(kernel='rbf', verbose=True, probability=True)
        clf.fit(train_vecs, y_train)
        joblib.dump(clf, 'svm_data/svm_model/model.pkl')
        # print(test_vecs)
        predict_y = clf.predict(test_vecs)  # 基于SVM对验证集做出预测，prodict_y 为预测的结果
        test_accuracy = metrics.accuracy_score(y_test, predict_y)  # 验证集上的准确率

        # y_pred = clf.predict(y_test)
        # test_precision = metrics.precision_score(y_test, y_pred, average='weighted')
        # test_recall = metrics.recall_score(y_test, y_pred, average='weighted')
        log.console_out("log_svm.txt", "SVM score = ", clf.score(test_vecs, y_test))  # 记录得分
        log.console_out("log_svm.txt", "test_accuracy = ", test_accuracy)  # 记录准确率

    # 得到待预测单个句子的词向量
    @classmethod
    def get_predict_vecs(cls, words):
        n_dim = 128
        comment_w2v = Word2Vec.load('svm_data/w2v_model/w2v_model.pkl')
        # comment_w2v.train(words)
        train_vecs = cls.build_wordvector(words, n_dim, comment_w2v)
        # print train_vecs.shape
        return train_vecs

    # 对单个句子进行情感判断
    @classmethod
    def predict(cls, string):
        words = cls.text_parse(string)
        words_vecs = cls.get_predict_vecs(words)
        clf = joblib.load('svm_data/svm_model/model.pkl')

        result = clf.predict(words_vecs)
        return result[0]


SVMClassifer.train()
res = SVMClassifer.predict('I hate you,I JUST THINK THIS THING IS BAD')
print(res)
