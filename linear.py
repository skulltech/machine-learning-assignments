import argparse
import sys

import numpy as np
from sklearn.linear_model import LassoLars
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures


def linear_regression(args):
    with open(args.trainfile) as f:
        train = np.genfromtxt(f, delimiter=',')
    x = np.column_stack((np.ones(train.shape[0], ), train[:, :-1]))
    y = train[:, -1]

    w = np.linalg.inv(x.T @ x) @ (x.T @ y)

    with open(args.testfile) as f:
        test = np.genfromtxt(f, delimiter=',')
    x = np.column_stack((np.ones(test.shape[0], ), test))
    predictions = x @ w
    np.savetxt(args.outputfile, predictions)
    np.savetxt(args.weightfile, w)


def ridge_regression(args):
    with open(args.trainfile) as f:
        train = np.genfromtxt(f, delimiter=',')
    x = np.column_stack((np.ones(train.shape[0], ), train[:, :-1]))
    y = train[:, -1]
    with open(args.regularization) as f:
        ls = np.genfromtxt(f, delimiter=',')

    kf = KFold(n_splits=10)
    kf.get_n_splits(x)
    min_error = float('inf')
    min_l = None

    for l in ls:
        error_sum = 0
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            w = np.linalg.inv((x_train.T @ x_train) + l * np.identity(x_train.shape[1])) @ (x_train.T @ y_train)
            error = np.linalg.norm((x_test @ w) - y_test) / (2 * (x.shape[0] // 10))
            error_sum += error
        if error_sum < min_error:
            min_error = error_sum
            min_l = l

    print(min_l)
    w = np.linalg.inv((x.T @ x) + min_l * np.identity(x.shape[1])) @ (x.T @ y)

    with open(args.testfile) as f:
        test = np.genfromtxt(f, delimiter=',')
    x = np.column_stack((np.ones(test.shape[0], ), test))
    predictions = x @ w
    np.savetxt(args.outputfile, predictions)
    np.savetxt(args.weightfile, w)


def feature_expansion(x):
    square = np.array(list(map(lambda i: i ** 2, x.T))).T
    cube = np.array(list(map(lambda i: i ** 2, x.T))).T
    mod = np.array(list(map(abs, x.T))).T
    x = np.hstack((x, square, cube, mod))
    return x


def lasso_regression(args):
    with open(args.trainfile) as f:
        train = np.genfromtxt(f, delimiter=',')
    x = feature_expansion(train[:, :-1])
    x = np.column_stack((np.ones(x.shape[0], ), x))
    y = train[:, -1]
    print(x.shape)

    kf = KFold(n_splits=10)
    kf.get_n_splits(x)
    ls = [0.001, 0.003, 0.005, 0.01]
    min_error = float('inf')
    min_l = None

    for l in ls:
        error_sum = 0
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            reg = LassoLars(alpha=l)
            reg.fit(x_train, y_train)
            error = np.linalg.norm(reg.predict(x_test) - y_test) / (2 * x_test.shape[0])
            error_sum += error
        print('l :', l, '. Error: ', error_sum)
        if error_sum < min_error:
            min_error = error_sum
            min_l = l

    print(min_l)
    reg = LassoLars(alpha=min_l)
    reg.fit(x, y)
    w = reg.coef_
    error = (np.linalg.norm(reg.predict(x) - y) ** 2) / (2 * x.shape[0])
    print(error)

    with open(args.testfile) as f:
        test = np.genfromtxt(f, delimiter=',')
    x = feature_expansion(test)
    x = np.column_stack((np.ones(test.shape[0], ), x))
    predictions = x @ w
    np.savetxt(args.outputfile, predictions)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Mode')

    a_parser = subparsers.add_parser('a', help='Linear regression using closed form solution.')
    a_parser.add_argument('trainfile', type=str)
    a_parser.add_argument('testfile', type=str)
    a_parser.add_argument('outputfile', type=str)
    a_parser.add_argument('weightfile', type=str)
    a_parser.set_defaults(func=linear_regression)

    b_parser = subparsers.add_parser('b', help='Ridge regression.')
    b_parser.add_argument('trainfile', type=str)
    b_parser.add_argument('testfile', type=str)
    b_parser.add_argument('regularization', type=str)
    b_parser.add_argument('outputfile', type=str)
    b_parser.add_argument('weightfile', type=str)
    b_parser.set_defaults(func=ridge_regression)

    c_parser = subparsers.add_parser('c', help='Regression with feature selection using Lasso Regression.')
    c_parser.add_argument('trainfile', type=str)
    c_parser.add_argument('testfile', type=str)
    c_parser.add_argument('outputfile', type=str)
    c_parser.set_defaults(func=lasso_regression)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
