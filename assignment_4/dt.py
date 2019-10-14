import argparse
import os
import sys
import time
import datetime

import numpy as np

CONTINOUS_COLUMNS = [0, 2, 3, 9, 10, 11]



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
    # print(x.shape)
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
        # print('[*] Leaf node.')
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
    if not nb:
        tree.children = []
        return 1
    if not tree.children:
        return 1
    
    nb = nb - 1
    for child in tree.children:
        prune(tree, nb)


def optimize(tree, x, y):
    best_tree = tree
    best_accr = predict(x, y)[1]
    
    while True:
        pass



def dt(args):
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

    preds, accuracy = predict(tree, x, y)
    np.savetxt(args.validpred, preds, fmt='%i')
    print(f'[*] Accuracy on validation data: {accuracy*100} %')


    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=',', dtype=object)
    test = np.delete(test, 3, 1)
    x = test[1:, :-1]
    with open(args.testres) as f:
        y = np.loadtxt(f, delimiter=',', dtype=int)
    
    preds, accuracy = predict(tree, x, y)
    np.savetxt(args.validpred, preds, fmt='%i')
    print(f'[*] Accuracy on test data: {accuracy*100} %')

    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('validfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('validpred', type=str)
    parser.add_argument('testpred', type=str)
    parser.add_argument('testres', type=str)
    parser.set_defaults(func=dt)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__=='__main__':
    main()
