# -*- coding: utf-8 -*-
import json
import time

import snownlp
import requests

from core import (
    SVMClassifer, LSTMClassifier)


class SentencePrediction:
    boson_nlp_url = 'http://api.bosonnlp.com/sentiment/analysis'
    baidu_nlp_url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token={}'

    @classmethod
    def get_sentiment_result(cls, sentence, method='default'):
        pass

    @classmethod
    def get_sentiment_result_by_lstm(cls, sentence, usage='three'):
        return LSTMClassifier.predict(sentence, usage)

    @classmethod
    def get_sentiment_result_by_svm(cls, sentence):
        return SVMClassifer.predict(sentence)

    @classmethod
    def get_sentiment_result_by_snownlp(cls, sentence):
        return snownlp.SnowNLP(sentence).sentiments

    @classmethod
    def get_sentiment_result_by_dict(cls, sentence):
        return

    @classmethod
    def get_sentiment_result_by_bosonnlp(cls, sentences):
        data = json.dumps(sentences)
        # r61YHEBH.9394.E5H8qTTOkEMy XJd34mJx.22007.gXtZWqKA3Kn9
        headers = {'X-Token': 'XJd34mJx.22007.gXtZWqKA3Kn9'}
        resp = requests.post(cls.boson_nlp_url, headers=headers, data=data.encode('utf-8'))
        try:
            items = resp.json()
            for i, item in enumerate(items):
                if item[0] < 0.45:
                    print(sentences[i])
        except (AttributeError, KeyError, json.decoder.JSONDecodeError):
            pass
        print(resp.text)
        return resp.text

    @classmethod
    def get_sentiment_result_by_baidunlp(cls, sentence):
        headers = {'Content-Type': 'application/json'}

        # TODO 用一个装饰器缓存token，只有当过期才再次请求
        def get_baidu_access_token(re_requset=False):
            with open('expire', 'r+') as f:
                token_endtime = f.read().split(' ')
            if len(token_endtime) == 2 and int(time.time()) < int(token_endtime[1]) and not re_requset:
                return token_endtime[0]

            api_key = '3z2ctjv7k3ELreUzfo7SOrjX'
            secret_key = 'DCf2oYx3YGYRGfUigZU2Zvf8c7uDUaqu'
            url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&' \
                  'client_secret={}'.format(api_key, secret_key)
            cont = requests.get(url, headers=headers).json()
            try:
                end_time = int(time.time()) + int(cont.get('expires_in'))
                token = cont.get('access_token')
                with open('expire', 'w') as fw:
                    fw.write('{} {}'.format(token, end_time))
                return token
            except (KeyError, AttributeError):
                print('本次请求百度NLP接口失败')
                return ''

        baidu_nlp_url = cls.baidu_nlp_url.format(get_baidu_access_token())
        data = json.dumps(dict(text=sentence))
        resp = requests.post(url=baidu_nlp_url, headers=headers, data=data.encode('utf8'))
        print(resp.text)
        # 2 stands for positive, 1 stands for neutual, 0 stands for negtive
        try:
            return int(resp.json().get('items')[0].get('sentiment'))
        except:
            return 1


class Checker:
    @classmethod
    def check_corpus_by(cls):
        total_count = 401
        limit = 100
        times = int(total_count / limit)

        cur_time = 0
        cur_pos = 0
        while cur_time <= times:
            if cur_time == times:
                limit = total_count - (limit * cur_time)

            sents = list()
            cur_count = 0
            with open('./dataset/hotel/pos.txt', 'r') as f:
                f.seek(cur_pos, 0)
                line = f.readline()
                sents.append(line)
                cur_count += 1
                while cur_count < limit:
                    cur_pos = f.tell()  # returns the location of the next line
                    line = f.readline()
                    sents.append(line)
                    cur_count += 1
            cur_time += 1
            SentencePrediction.get_sentiment_result_by_bosonnlp(sents)

    @classmethod
    def check_by_baidunlp(cls):
        with open('./dataset/neg.txt', 'r') as f:
            sentences = f.readlines()

        new_neg = 'pos2.txt'
        new_neu = 'neu2.txt'
        with open(new_neg, 'a+') as fneg, open(new_neu, 'a+') as fneu:
            for i, sentence in enumerate(sentences[14238:]):
                res = SentencePrediction.get_sentiment_result_by_baidunlp(sentence)

                if res == 2:
                    fneg.write(sentence)

                if res == 1:
                    fneu.write(sentence)

                with open('proc.txt', 'w') as pf:
                    pf.write(str(i))
                    pf.write(sentence)


if __name__ == '__main__':
    # string1 = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string2 = '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string3 = '我是傻逼'
    # string4 = '你是傻逼'
    # string5 = '屏幕较差，拍照也很粗糙。'
    # string6 = '辣鸡手机'
    # string7 = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # string8 = '酒店的环境非常好，价格也便宜，值得推荐'
    # string9 = '质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    # string10 = '东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    # print(SentencePrediction.get_sentiment_result_by_lstm(string1))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string2))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string3))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string4))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string5))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string6))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string7))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string8))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string9))
    # print(SentencePrediction.get_sentiment_result_by_lstm(string10))
    # print(SentencePrediction.get_sentiment_result_by_lstm('他是个坏蛋，很爱做恶作剧'))
    # print(SentencePrediction.get_sentiment_result_by_lstm('她是个善良的女孩，温柔可爱'))
    # print(SentencePrediction.get_sentiment_result_by_lstm('今天天气不错，让我心情舒畅'))
    # print(SentencePrediction.get_sentiment_result_by_lstm('我真的很伤心啊'))
    print(SentencePrediction.get_sentiment_result_by_lstm('三清山索道。门票坑爹​'))














