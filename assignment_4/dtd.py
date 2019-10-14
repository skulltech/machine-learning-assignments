import argparse
import os
import sys
import time
import datetime
from copy import deepcopy

import numpy as np

CONTINOUS_COLUMNS = [0, 2, 3, 9, 10, 11]
TTL = 30


class Node:
    def __init__(self, prediction, continuous=None, unqs=None, column=None, median=None):
        self.children = []
        self.column = column
        self.continuous = continuous
        self.unqs = unqs
        self.median = median
        self.prediction = prediction


def entropy(xd, y, continuous, label=None):
    indicesl = []
    median = None
    unqs = None

    if not continuous:
        unqs, counts = np.unique(xd, return_counts=True)
        entropy = 0

        for unq, count in zip(unqs, counts):
            indices = np.argwhere(xd == unq)
            indicesl.append(indices)
            ys = y[indices]
            cnts = np.unique(ys, return_counts=True)[1]
            probs = cnts / ys.shape[0]
            ent = np.sum(-1 * probs * np.log2(probs))
            entropy = entropy + ((count / xd.shape[0]) * ent)    
    else:
        xd = xd.astype(int)
        median = np.median(xd)
        
        entropy = 0
        conds = [xd < median, xd >= median]
        for cond in conds:
            indices = np.argwhere(cond)
            indicesl.append(indices)
            ys = y[indices]
            cnts = np.unique(ys, return_counts=True)[1]
            probs = cnts / ys.shape[0]
            ent = np.sum(-1 * probs * np.log2(probs))
            entropy = entropy + ((ys.shape[0] / xd.shape[0]) * ent) 
    
    # if label: print(label, entropy)
    return entropy, indicesl, median, unqs


def create_tree(x, y, labels):
    # print(x.shape[0], 'rows.')
    ents = []
    indicesll = []
    medians = []
    unqsl = []

    for i in range(x.shape[1]):
        ent, indicesl, median, unqs = entropy(x[:, i], y, continuous=i in CONTINOUS_COLUMNS, label=labels[i])
        ents.append(ent)
        indicesll.append(indicesl)
        medians.append(median)
        unqsl.append(unqs)

    minent = min(ents)
    vals, cnts = np.unique(y, return_counts=True)
    prediction = vals[np.argmax(cnts)]
    
    if not minent or len(list(filter(lambda x: x.shape[0] > 0, indicesl))) < 2:
        # print('Leaf node.')
        node = Node(prediction=prediction)
        return node

    column = ents.index(minent)
    indicesl = indicesll[column]
    median = medians[column]
    unqs = unqsl[column]

    # print('[*] Splitting by column', column, ':', labels[column])
    # print('[*] Number of branches :', len(indicesl))

    node = Node(prediction=prediction, column=column, continuous=column in CONTINOUS_COLUMNS, median=median, unqs=unqs)
    for indices in indicesl:
        indices = indices.flatten()
        child = create_tree(x[indices, :], y[indices, :], labels)
        node.children.append(child)
    
    if len(node.children) < 2:
        node.children = []
        node.column = None
        node.median = None
    
    return node


def height(tree):
    return 1 + max([height(child) for child in tree.children]) if tree.children else 1


def cnodes(tree):
    return 1 + sum([cnodes(child) for child in tree.children])


def __predict(tree, xr):
    if not tree.children: return tree.prediction

    if tree.continuous:
        if int(xr[tree.column]) < tree.median:
            return __predict(tree.children[0], xr)
        else:
            return __predict(tree.children[1], xr)
    else:
        try:
            return __predict(tree.children[list(tree.unqs).index(xr[tree.column])], xr)
        except ValueError:
            return tree.prediction


def predict(tree, x, y=None):
    preds = []
    accuracy = None

    for i in range(x.shape[0]):
        preds.append(__predict(tree, x[i, :]))
    preds = np.array(preds)

    if isinstance(y, np.ndarray):
        y = y.flatten().astype(np.uint8)
        accuracy = np.sum(preds == y) / y.shape[0]
    return preds, accuracy



def prune(tree, nb):
    copied = deepcopy(tree)
    count = 0
    stack = [copied]

    while True:
        node = stack.pop()
        if count == nb:
            # print('Node nb', nb, ', Removing', len(node.children), 'children.')
            node.children = []
            return copied
        for child in node.children:
            stack.append(child)
        count = count + 1


def optimize(tree, x, y, begin):
    global_best_tree = tree
    global_best_accr = predict(tree, x, y)[1]

    while True:
        start = time.time()
        best_tree = global_best_tree
        best_accr = global_best_accr
        print(height(global_best_tree), cnodes(global_best_tree), global_best_accr)

        for i in range(cnodes(global_best_tree)):
            if time.time() - begin > TTL:
                return best_tree

            pruned = prune(global_best_tree, i)
            # print(f'[*] Pruned node {i}. Height: {height(pruned)}. Nodes: {cnodes(pruned)}.')
            accr = predict(pruned, x, y)[1]
            if accr > best_accr:
                best_accr = accr
                best_tree = pruned
        print('[*] Iteration time:', time.time() - start)
        if best_accr > global_best_accr:
            global_best_accr = best_accr
            global_best_tree = best_tree
        else:
            return global_best_tree



def dt(args):
    begin = time.time()
    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=',', dtype=object)
    train = np.delete(train, 3, 1)
    x = train[1:, :-1]
    y = train[1:, -1:]
    y = y.astype(np.uint8)
    labels = train[0, :]

    tree = create_tree(x, y, labels)
    print(f'[*] Tree created. Height: {height(tree)}. Nodes: {cnodes(tree)}.')

    
    with open(args.validfile) as f:
        valid = np.loadtxt(f, delimiter=',', dtype=object)
    valid = np.delete(valid, 3, 1)
    x = valid[1:, :-1]
    y = valid[1:, -1:]

    optimized = optimize(tree, x, y, begin)
    print(f'[*] Optimized tree. Height: {height(optimized)}. Nodes: {cnodes(optimized)}.')

    preds, accuracy = predict(optimized, x, y)
    np.savetxt(args.validpred, preds, fmt='%i')
    print('[*] Accuracy on validation data:', accuracy)


    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=',', dtype=object)
    test = np.delete(test, 3, 1)
    x = test[1:, :-1]
    if args.testlabels:
        with open(args.testlabels) as f:
            y = np.loadtxt(f, delimiter=',', dtype=int)
        preds, accuracy = predict(optimized, x, y)
        print('[*] Accuracy on test data:', accuracy)
    else:
        preds, accuracy = predict(optimized, x)
    np.savetxt(args.testpred, preds, fmt='%i')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('validfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('validpred', type=str)
    parser.add_argument('testpred', type=str)
    parser.add_argument('testlabels', nargs='?', default='', type=str)
    parser.set_defaults(func=dt)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__=='__main__':
    main()
