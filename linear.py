import argparse
import sys

import numpy as np


def linear_regression(args):
    with open(args.trainfile) as f:
        arr = np.genfromtxt(f, delimiter=',')
    x = arr[:, :-1]
    y = arr[:, -1]
    xinv = np.linalg.pinv(x)
    w = np.matmul(xinv, y)
    print(len(w))


def ridge_regression(args):
    pass


def lasso_regression(args):
    pass


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
    parser.parse_args()


if __name__ == '__main__':
    main()
