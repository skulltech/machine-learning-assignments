import numpy as np
import sys
import argparse
import math
import time


def one_hot_encode_target(y):
    ylabels = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']
    yd = np.zeros((y.shape[0], 5))
    for i in range(y.shape[0]):
        yd[i, ylabels.index(y[i])] = 1
    return yd


def one_hot_encode_features(x):    
    xlabels = [['usual', 'pretentious', 'great_pret'],
    ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
    ['complete', 'completed', 'incomplete', 'foster'],
    ['1', '2', '3', 'more'],
    ['convenient', 'less_conv', 'critical'],
    ['convenient', 'inconv'],
    ['nonprob', 'slightly_prob', 'problematic'],
    ['recommended', 'priority', 'not_recom']]
    xd = np.zeros((x.shape[0], 27))
    for i in range(x.shape[0]):
        start = 0
        for j in range(x.shape[1]):
            xd[i, start + xlabels[j].index(x[i, j])] = 1
            start = start + len(xlabels[j])
    return xd


def cross_entropy(y, predictions, eps=1e-15):
    loss = lambda y, p: np.sum(-1 * y * np.log(np.clip(p, eps, 1-eps)), axis=1)
    l = loss(y, predictions)
    cost = np.sum(l) / l.shape[0] 
    return cost


def softmax(x):
    x = np.exp(x)
    return x / np.sum(x, axis=0)


def predict(x, w):
    lw = x @ w
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    predictions = softmax(lw.T).T
    return predictions


def update_weights(x, y, w, lr):
    predictions = predict(x, w)
    gradient = (x.T @ (predictions - y)) / x.shape[0]
    w = w - (gradient * lr)
    return w


constant_lr = lambda x: float(x)
adaptive_lr = lambda x, i: float(x) / math.sqrt(i+1)

def backtracking_line_search(x, y, w, alpha, beta, eta):
    gradient = (x.T @ (predict(x, w) - y)) / x.shape[0]
    f = lambda w: cross_entropy(y, predict(x, w))
    fw = f(w)
    while (f(w - eta*gradient) > fw - alpha*eta*(np.linalg.norm(gradient)**2)):
        eta = eta * beta
    return eta

    

def logistic_regression(args):
    with open(args.trainfile) as f:
        train = np.genfromtxt(f, delimiter=',', dtype=str)
    x = one_hot_encode_features(train[:, :-1])
    x = np.column_stack((np.ones(train.shape[0]), x))
    y = one_hot_encode_target(train[:, -1])
    w = np.zeros((x.shape[1], y.shape[1]))

    with open(args.param) as f:
        lines = f.readlines()
    strategy = int(lines[0])
    lr_params = lines[1]
    iters = int(lines[2])
    batch_size = int(lines[3])
    
    inner_loops = x.shape[0] // batch_size
    for i in range(iters):
        if strategy == 1:
            lr = constant_lr(lr_params)
        if strategy == 2:
            lr = adaptive_lr(lr_params, i)
        if strategy == 3:
            if type(lr_params) == str:
                lr_params = lr_params.split(',')
            eta = float(lr_params[0].strip())
            alpha = float(lr_params[1].strip())
            beta = float(lr_params[2].strip())
            lr = backtracking_line_search(x, y, w, alpha, beta, eta)

        for j in range(inner_loops):
            xd = x[j*batch_size:(j+1)*batch_size, :]
            yd = y[j*batch_size:(j+1)*batch_size, :]
            w = update_weights(xd, yd, w, lr)

        # print(f'Iteration: {i+1}\t Learning rate: {lr}\t Cost: {cross_entropy(y, predict(x, w))}')

    # print(f'Final cost: {cross_entropy(y, predict(x, w))}')

    with open(args.testfile) as f:
        test = np.genfromtxt(f, delimiter=',', dtype=str)
    
    np.savetxt(args.weightfile, w, delimiter=',')
    x = one_hot_encode_features(test)
    x = np.column_stack((np.ones(test.shape[0]), x))
    predictions = predict(x, w)

    final_predictions = np.ndarray((x.shape[0], ), dtype='U32')
    ylabels = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']
    for i in range(x.shape[0]):
        maxloc = np.argmax(predictions[i])
        final_predictions[i] = ylabels[maxloc]
    np.savetxt(args.outputfile, final_predictions, fmt='%s')
    print(iters)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('param', type=str)
    parser.add_argument('outputfile', type=str)
    parser.add_argument('weightfile', type=str)
    parser.set_defaults(func=logistic_regression)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__ == '__main__':
    main()
