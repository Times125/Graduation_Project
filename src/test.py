#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lichunhui
@Time:  2018/4/10 13:01
@Description: 
"""

import nltk
import re


def text_parse(x):
    try:
        sentence = x.strip().lower()
    except:
        sentence = x
    sentence = re.sub(r'@\s*[\w]+ | ?#[\w]+ | ?&[\w]+; | ?[^\x00-\xFF]+', '', sentence)
    pattern = r""" (?x)(?:[a-z]\.)+ 
                       | \d+(?:\.\d+)?%?\w+
                       | \w+(?:[-']\w+)*
                       | <\d+
                       | (?:[><*$]+[._]*[<>*$]+)
                       | [:;][\w*\d*][-]*[)(]*
                       | (?:[-.!?]{2,})
                       | [][.,;"'?():$-_*`]"""
    word_list = nltk.regexp_tokenize(sentence, pattern)
    print(word_list)


if __name__ == '__main__':
    x = "still misses  him  and worries like it's her job or something.. ????? ------  >_<  >_> >< *_* :D ;0) Work's fine? an interesting coworker, ha! o_O Lots of typing, what fun! <3<3"
    text_parse(x)
