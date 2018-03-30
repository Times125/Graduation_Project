from core import (
    SVMClassifer, LSTMClassifier)


class Trainer:
    pos_path = './dataset/three_classify/pos.txt'
    neu_path = './dataset/three_classify/neu.txt'
    neg_path = './dataset/three_classify/neg.txt'

    @classmethod
    def svm_train(cls):
        SVMClassifer.train()

    @classmethod
    def lstm_two_classify_train(cls):
        LSTMClassifier.train(cls.pos_path, cls.neg_path)

    @classmethod
    def lstm_three_classify_train(cls):
        LSTMClassifier.train(cls.pos_path, cls.neu_path, cls.neg_path)

    @classmethod
    def lstm_train(cls):
        LSTMClassifier.train()


if __name__ == '__main__':
    # Trainer.lstm_two_classify_train()
    Trainer.lstm_three_classify_train()
