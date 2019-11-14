import argparse
import sys

import numpy as np


class Classifier:
    def __init__(self, x, y, c, lamb):
        self.x = x
        self.y = y
        self.c = c
        self.lamb = lamb
        self.cs = {}
        for i in range(10):
            for j in range(i + 1, 10):
                indices = np.argwhere(np.logical_or(y == i, y == j))
                indices = indices.flatten()
                xs = x[indices]
                ys = y[indices]
                ys = -1 * (ys == i) + 1 * (ys == j)
                self.cs[(i, j)] = SVM(xs, ys, i, j, c, lamb)

    def train(self, batch_size, iterations):
        for k, v in self.cs.items():
            for i in range(iterations):
                v.train(batch_size, i + 1)

    def predict(self, x):
        ps = []
        for k, v in self.cs.items():
            ps.append(v.predict(x))
        ps = np.asarray(ps)

        def func(x):
            uniques, counts = np.unique(x, return_counts=True)
            return uniques[np.argmax(counts)]

        ps = np.apply_along_axis(func, 0, ps)
        return ps


class SVM:
    def __init__(self, x, y, label1, label2, c, lamb):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1],))
        self.b = 0
        self.c = c
        self.lamb = lamb
        self.label1 = label1
        self.label2 = label2

    def train(self, batch_size, iteration):
        indices = np.random.choice(self.x.shape[0], batch_size, replace=False)
        xt = self.x[indices]
        yt = self.y[indices]
        ip = np.argwhere(yt * (xt @ self.w) < 1).flatten()
        xtp = xt[ip]
        ytp = yt[ip]
        eta = 1 / (self.lamb * iteration)
        self.w = ((1 - eta * self.lamb) * self.w) + (eta * self.c / batch_size) * np.sum((xtp.T * ytp).T, axis=0)
        self.b = self.b + (eta * self.c / batch_size) * np.sum(ytp)

    def predict(self, x):
        v = x @ self.w + self.b
        ps = self.label1 * (v <= 0) + self.label2 * (v > 0)
        return ps


def svm(args):
    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=',', dtype=np.uint8)

    x = train[:, :-1]
    x = x / 255
    y = train[:, -1]
    classifier = Classifier(x, y, c=0.0014, lamb=0.001)
    classifier.train(batch_size=1000, iterations=40)

    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=',', dtype=np.uint8)
    x = test[:, :-1]
    x = x / 255
    ps = classifier.predict(x)
    if args.testlabels:
        with open(args.testlabels) as f:
            y = np.loadtxt(f, delimiter=',', dtype=np.uint8)
        accuracy = np.sum(ps == y) / y.shape[0]
        print('Accuracy on test data:', accuracy)
    np.savetxt(args.testpred, ps, fmt='%i')


def main():
    parser = argparse.ArgumentParser()
    np.random.seed(10)

    parser.add_argument('trainfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('testpred', type=str)
    parser.add_argument('testlabels', nargs='?', default='', type=str)
    parser.set_defaults(func=svm)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
