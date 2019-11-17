import argparse
import string
import sys

import numpy as np


class NaiveBayes:
    def __init__(self):
        self.dict0 = {}
        self.dict1 = {}
        self.count0 = 0
        self.count1 = 0
        self.samples = 0
        self.vocabulary = 0

    @staticmethod
    def process(r):
        return set(r.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).lower().split())

    def train(self, reviews):
        for r, l in reviews:
            if not l:
                self.count0 = self.count0 + 1
            else:
                self.count1 = self.count1 + 1
            ws = self.process(r)
            for word in ws:
                if not l:
                    self.dict0[word] = self.dict0[word] + 1 if word in self.dict0 else 1
                else:
                    self.dict1[word] = self.dict1[word] + 1 if word in self.dict1 else 1
        self.samples = len(reviews)
        self.vocabulary = len({**self.dict0, **self.dict1})

    def predict(self, review):
        review = self.process(review)
        p0 = np.log(self.count0 / self.samples)
        p1 = np.log(self.count1 / self.samples)
        for word in review:
            p0 = p0 + np.log((self.dict0.get(word, 0) + 1) / (self.count0 + self.vocabulary + 1))
            p1 = p1 + np.log((self.dict1.get(word, 0) + 1) / (self.count1 + self.vocabulary + 1))
        return 0 if p0 > p1 else 1


def naive_bayes(args):
    with open(args.trainfile, 'r', encoding='UTF-8') as f:
        train = f.readlines()
    train = train[1:]

    reviews = []
    for i, line in enumerate(train):
        split = line.split(',')
        label = int(split[-1].strip() == 'positive')
        review = ','.join(split[:-1])[1:-1]
        reviews.append((review, label))

    model = NaiveBayes()
    model.train(reviews)
    print('[*] Training complete.')

    with open(args.testfile, 'r', encoding='UTF-8') as f:
        test = f.readlines()
    test = test[1:]

    x = [line[1:-2] for line in test]
    y = np.zeros((len(test),))

    for i, review in enumerate(x):
        y[i] = model.predict(review)
    np.savetxt(args.outputfile, y, fmt='%i')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.set_defaults(func=naive_bayes)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
