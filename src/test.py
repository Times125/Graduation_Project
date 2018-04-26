#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lichunhui
@Time:  2018/4/10 13:01
@Description: 
"""

import nltk
import re
from src import log
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier


class Testme:
    # 去除类似soooo 保留soo
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)

    # twitter中@***这种将被替换成__HAND_
    hndl_regex = re.compile(r"@(\w+)")

    # twitter中话题#***这种将被替换成__HASH_
    hash_regex = re.compile(r"#(\w+)")

    # twitter中话题#***这种将被替换成__URL_
    url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

    # twitter中存在的一些表情
    emoticons = [('__EMOT_SMILEY', [':-)', ':)', '(:', '(-:', '^_^', '>y<', '>o<', '>O<', '^.^']),
                 ('__EMOT_LAUGH', [':-D', ':D', 'X-D', 'XD', 'xD', ]),
                 ('__EMOT_LOVE', ['<3', ':\*', ]),
                 ('__EMOT_WINK', [';-)', ';)', ';-D', ';D', '(;', '(-;', '←_←', '→_→', '<_<', '>_>']),
                 ('__EMOT_FROWN', [':-(', ':(', '(:', '(-:', '=_=', 'D:<','orz']),
                 ('__EMOT_CRY', [':,(', ':\'(', ':"(', ':((', 't_t', 'T_T', '→_←', 'T^T', '>_<']),
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
         wont|wouldnt|dont|doesnt|didnt|isnt|aret|aint)$)|n't""", re.X)

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


if __name__ == '__main__':
    #log.console_out('1.txt',(1,2,(4,5)))
    #log.console_out('2.txt','22222')
    # x = "@times doesn't miss  him  and worries like it's her job or something #HMS.. ????? ------  >_<  >_> >< *_* :D ;0) Work's fine? an interesting coworker, ha! o_O Lots of typing, what fun! <3<3"
    x = '@MrsFox Yeah...doesn\'t taste like #bacon at all :-(  http://tinyurl.com/c2wwgp'
    # feat = {}
    # x = "@times still doesn't miss  him  and worries like it's ? .her job or something ??? .... ---------"
    y = Testme.text_parse(x)
    print(y)
    # print('\n')
    # negation_features = Testme.get_negation_features(y)
    # print("negation_features-->", negation_features)
    # print('\n')
    # words_bi = ['has(%s)' % ','.join(map(str, bg)) for bg in nltk.bigrams(y)]
    # print(words_bi)
    # print('\n')
    # bag = {}
    # for f in words_bi:
    #     bag[f] = 1
    # feat.update(bag)
    # feat.update(negation_features)
    # print(feat)
    # classifier = SklearnClassifier(BernoulliNB())
    # classifier.train([[feat, 'neg']])
    # print(classifier.classify(feat))
