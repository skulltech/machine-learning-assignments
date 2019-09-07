import numpy as np
import argparse
import sys
import math
import time
from sklearn.preprocessing import LabelBinarizer
from skimage import feature



HYPERPARAMS = {
    'init_factor': 0.01,
    'activation_fn': 'sigmoid',
    'base_lr': 1,
    'batch_size': 100,
    'hidden_layers': [40],
    'time': 100
}


def softmax(x):
    x = x - np.max(x, axis=1)[:, None]
    x = np.exp(x)
    sm = x / np.repeat(np.sum(x, axis=1)[:, None], x.shape[1], axis=1)
    return sm


sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)
rectifier = lambda x: np.maximum(x, 0)
drectifier = lambda x: np.greater(x, 0).astype(float)

ACTIVATION = {
    'sigmoid': {
        'function': sigmoid,
        'derivative': dsigmoid
    },
    'rectifier': {
        'function': rectifier,
        'derivative': drectifier
    }
}


def cross_entropy(y, probs, eps=1e-15):
    loss = lambda y, p: np.sum(-1 * y * np.log(np.clip(p, eps, 1-eps)), axis=1)
    l = loss(y, probs)
    cost = np.sum(l) / l.shape[0] 
    return cost


def accuracy(y, preds):
    return np.sum(y.flatten() == preds) * 100 / y.shape[0] 


class NeuralNetwork:
    def __init__(self, hidden_layers, feature_size, output_size):
        prev = feature_size
        self.weights = []
        self.biases = []
        factor = HYPERPARAMS['init_factor']

        for x in hidden_layers:
            self.weights.append(np.random.rand(prev, x) * factor)
            prev = x
        self.weights.append(np.random.rand(prev, output_size) * factor)

        for x in hidden_layers:
            self.biases.append(np.random.rand(1, x) * factor)
        self.biases.append(np.random.rand(1, output_size) * factor)
    
    def forprop(self, x):
        activs = [x]
        prev = x
    
        for weight, bias in zip(self.weights, self.biases):
            z = prev @ weight + np.repeat(bias, x.shape[0], axis=0)
            activ = ACTIVATION[HYPERPARAMS['activation_fn']]['function'](z)
            activs.append(activ)
            prev = activ
        
        activs[-1] = softmax(z)
        return activs
    
    def predict(self, x):
        probs = self.forprop(x)[-1]
        preds = np.argmax(probs, axis=1)
        return probs, preds

    def backprop(self, y, activs, lr):
        pred = activs[-1]
        delta = (pred - y)
        
        i = len(self.weights) - 1
        for weight, bias, activ in zip(reversed(self.weights), reversed(self.biases), reversed(activs[:-1])):
            dW = activ.T @ delta
            dB = np.sum(delta, axis=0, keepdims=True)
            delta = (delta @ weight.T) * ACTIVATION[HYPERPARAMS['activation_fn']]['derivative'](activ)
            self.weights[i] = weight - (lr * dW / y.shape[0])
            self.biases[i] = bias - (lr * dB / y.shape[0])
            i = i - 1

    def train(self, x, y, lr):
        activs = self.forprop(x)
        self.backprop(y, activs, lr)

    def dump(self, weightfile):
        output = np.concatenate([np.concatenate((bias.flatten(), weight.flatten())) for bias, weight in zip(self.biases, self.weights)])
        np.savetxt(weightfile, output)


def neural_network(args):
    start = time.time()

    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=',', dtype=np.uint8)
    x = train[:, :-1]
    ylabels = train[:, -1:]

    edge_detector = lambda x: feature.canny(x.reshape((32, 32))).flatten()
    x = np.hstack((x, np.apply_along_axis(edge_detector, 1, x)))
    x = x / 255
    
    lb = LabelBinarizer()
    y = lb.fit_transform(ylabels)

    test_samples = 1000
    xtrain = x[:-test_samples, :]
    ytrain = y[:-test_samples, :]
    xtest = x[-test_samples:, :]
    ytest = ylabels[-test_samples:, :]

    batch_size = HYPERPARAMS['batch_size']
    batches = xtrain.shape[0] // batch_size
    nn = NeuralNetwork(hidden_layers=HYPERPARAMS['hidden_layers'], feature_size=x.shape[1], output_size=y.shape[1])

    i = 0
    epoch = 0
    iter = 0
    best_wbs = None
    best_ac = -1 * float('inf')
    ascends = 0

    while True:
        lr = HYPERPARAMS['base_lr'] / math.sqrt(epoch + 1)

        xd = xtrain[batch_size*i:batch_size*(i+1), :]
        yd = ytrain[batch_size*i:batch_size*(i+1), :]
        nn.train(xd, yd, lr)
        i = i + 1
        if i >= batches:
            epoch = epoch + 1
            preds = nn.predict(xtest)[1]
            ac = accuracy(ytest, preds)

            if ac >= best_ac:
                best_wbs = nn.weights[:], nn.biases[:]
                best_ac = ac
            else:
                ascends = ascends + 1
            if ascends > 5:
                nn.weights, nn.biases = best_wbs[0][:], best_wbs[1][:]
                lr = lr / 2
                ascends = 0

            time_elapsed = time.time() - start
            print(f'Epoch: {epoch}\t Accuracy: {accuracy(ytest, preds)}\t Time: {time_elapsed}')
            if time_elapsed >  HYPERPARAMS['time']:
                break
        
        i = i % batches
        iter = iter + 1
    
    nn.weights, nn.biases = best_wbs[0][:], best_wbs[1][:]

    with open(args.testfile) as f:
        test = np.loadtxt(f, delimiter=',', dtype=np.uint8)
    x = test[:, :-1]
    x = np.hstack((x, np.apply_along_axis(edge_detector, 1, x)))
    x = x / 255
    
    preds = nn.predict(x)[1]
    np.savetxt(args.outputfile, preds, fmt='%i')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('testfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.set_defaults(func=neural_network)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__=='__main__':
    main()
