import sys
import multiprocessing

import yaml
import numpy as np
import jieba
import keras
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
    vocab_dim = 100
    maxlen = 100
    n_iterations = 1  # ideally more..
    n_exposures = 10
    window_size = 7
    batch_size = 32
    n_epoch = 10
    input_length = 100
    cpu_count = multiprocessing.cpu_count() or 4

    # 加载训练文件
    @classmethod
    def load_file(cls, pos_path, neg_path, neu_path=None, usage='two'):
        with open(pos_path, 'r') as f:
            pos = f.readlines()

        with open(neg_path, 'r') as f:
            neg = f.readlines()

        if usage == 'three':
            with open(neu_path, 'r') as f:
                neu = f.readlines()

            combined = np.concatenate((pos, neu, neg))
            y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int),
                                -1 * np.ones(len(neg), dtype=int)))
        else:
            combined = np.concatenate((pos, neg))
            y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
        return combined, y

    @classmethod
    def tokenizer(cls, text):
        text = [jieba.lcut(document.replace('\n', '')) for document in text]
        return text

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
            combined = sequence.pad_sequences(combined, maxlen=cls.maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
            return w2indx, w2vec, combined
        else:
            print('No data provided...')

    # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
    @classmethod
    def word2vec_train(cls, combined, usage):
        model = Word2Vec(size=cls.vocab_dim, min_count=cls.n_exposures, window=cls.window_size,
                         workers=cls.cpu_count, iter=cls.n_iterations)
        model.build_vocab(combined)
        model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
        model.save(W2V_MODEL_PATH.format(usage))
        index_dict, word_vectors, combined = cls.create_dictionaries(model=model, combined=combined)
        return index_dict, word_vectors, combined

    @classmethod
    def get_data(cls, index_dict, word_vectors, combined, y, usage='two'):
        n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, cls.vocab_dim))  # 索引为0的词语，词向量全为0
        for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = word_vectors[word]
        x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
        if usage == 'three':
            y_train = keras.utils.to_categorical(y_train, num_classes=3)
            y_test = keras.utils.to_categorical(y_test, num_classes=3)
        return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

    @classmethod
    def get_data_of_two_classify(cls, index_dict, word_vectors, combined, y):
        n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, cls.vocab_dim))  # 索引为0的词语，词向量全为0
        for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = word_vectors[word]
        x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
        return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

    @classmethod
    def train_lstm(cls, n_symbols, embedding_weights, x_train, y_train, x_test, y_test, usage='two'):
        model = Sequential()
        model.add(Embedding(output_dim=cls.vocab_dim, input_dim=n_symbols, mask_zero=True,
                            weights=[embedding_weights], input_length=cls.input_length))  # Adding Input Length

        if usage == 'three':
            model.add(LSTM(units=50, activation='tanh'))
            model.add(Dropout(0.5))
            model.add(Dense(3, activation='softmax'))  # Dense=>全连接层,输出维度=3
            model.add(Activation('softmax'))
        else:
            model.add(LSTM(units=50, activation='sigmoid', inner_activation='hard_sigmoid'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=cls.batch_size, epochs=cls.n_epoch,
                  verbose=1, validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, batch_size=cls.batch_size)

        yaml_string = model.to_yaml()
        with open(LSTM_YML_PATH.format(usage), 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights(LSTM_MODEL_PATH.format(usage))
        print('Test score:', score)

    @classmethod
    def three_classify(cls, pos_path, neg_path, neu_path):
        combined, y = cls.load_file(pos_path, neg_path, neu_path, 'three')
        combined = cls.tokenizer(combined)
        index_dict, word_vectors, combined = cls.word2vec_train(combined, 'three')
        n_symbols, embedding_weights, x_train, y_train, x_test, y_test = cls.get_data(index_dict, word_vectors,
                                                                                      combined, y, 'three')
        cls.train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, 'three')

    @classmethod
    def two_classify(cls, pos_path, neg_path):
        combined, y = cls.load_file(pos_path, neg_path)
        combined = cls.tokenizer(combined)
        index_dict, word_vectors, combined = cls.word2vec_train(combined, 'two')
        n_symbols, embedding_weights, x_train, y_train, x_test, y_test = cls.get_data(index_dict, word_vectors,
                                                                                      combined, y, 'two')
        cls.train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, 'two')

    # 训练模型，并保存
    @classmethod
    def train(cls, *args):
        if len(args) == 3:
            pos = args[0]
            nue = args[1]
            neg = args[2]
            cls.three_classify(pos, neg, nue)
        elif len(args) == 2:
            pos = args[0]
            neg = args[1]
            cls.two_classify(pos, neg)
        else:
            raise ValueError('You must input three or two file path')

    @classmethod
    def input_transform(cls, string, usage):
        words = jieba.lcut(string)
        words = np.array(words).reshape(1, -1)
        model = Word2Vec.load(W2V_MODEL_PATH.format(usage))
        _, _, combined = cls.create_dictionaries(model, words)
        return combined

    @classmethod
    def predict(cls, sentence, usage='two'):
        with open(LSTM_YML_PATH.format(usage), 'r') as f:
            yaml_string = yaml.load(f)
        model = model_from_yaml(yaml_string)
        model.load_weights(LSTM_MODEL_PATH.format(usage))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        data = cls.input_transform(sentence, usage)
        data.reshape(1, -1)
        # print data
        result = model.predict_classes(data)
        print(model.predict(data))

        # two classify 0 neg 1 pos; three classify -1 for neg 1 neu 2 pos;
        if usage == 'three':
            if result[0] == 2:
                return -1
            return result[0]
        else:
            return result[0][0]