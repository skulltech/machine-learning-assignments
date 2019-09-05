import numpy as np
import argparse
import sys
import math


class NeuralNetwork:
    def __init__(self, hidden_layers, feature_size, output_size):
        self.hidden_layers = hidden_layers
        prev = feature_size
        self.weights = []
        self.biases = []

        for x in hidden_layers:
            self.weights.append(np.zeros((prev, x)))
            prev = x
        self.weights.append(np.zeros((prev, output_size)))

        for x in hidden_layers:
            self.biases.append(np.zeros((1, x)))
        self.biases.append(np.zeros((1, output_size)))
    

    def forprop(self, x):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        activs = [x]
        prev = x
    
        for weight, bias in zip(self.weights, self.biases):
            z = prev @ weight + np.repeat(bias, x.shape[0], axis=0)
            activ = sigmoid(z)
            activs.append(activ)
            prev = activ
        return activs


    def backprop(self, y, activs, lr):
        pred = activs[-1]
        delta = (pred - y)
        modified_weights = []
        modified_biases = []
        
        for weight, bias, activ in (zip(reversed(self.weights), reversed(self.biases), reversed(activs[:-1]))):
            dW = activ.T @ delta
            dB = np.sum(delta, axis=0, keepdims=True)
            delta = (delta @ weight.T) * (1 - activ) * activ
            modified_weights.insert(0, weight - (lr * dW / y.shape[0]))
            modified_biases.insert(0, bias - (lr * dB / y.shape[0]))
        
        return modified_weights, modified_biases


    def train(self, x, y, lr):
        activs = self.forprop(x)
        self.weights, self.biases = self.backprop(y, activs, lr)
    

    def dump(self, weightfile):
        output = np.concatenate([np.concatenate((bias.flatten(), weight.flatten())) for bias, weight in zip(self.biases, self.weights)])
        np.savetxt(weightfile, output)


def neural_network(args):
    with open(args.trainfile) as f:
        train = np.genfromtxt(f, delimiter=',')
    x = train[:, :-1]
    y = train[:, -1:]

    with open(args.param) as f:
        param = f.readlines()
    strategy = int(param[0])
    lr = float(param[1])
    iterations = int(param[2])
    batch_size = int(param[3])
    hidden_layers = [int(x) for x in param[4].split()]

    batches = x.shape[0] // batch_size
    nn = NeuralNetwork(hidden_layers=hidden_layers, feature_size=x.shape[1], output_size=1)
    
    i = 0
    for iter in range(iterations):
        if strategy == 1:
            lr = lr
        else:
            lr = lr / math.sqrt(iter)
        
        xd = x[batch_size*i:batch_size*(i+1), :]
        yd = y[batch_size*i:batch_size*(i+1), :]
        nn.train(xd, yd, lr)
        i = (i + 1) % batches

    print(iterations)
    nn.dump(args.weightfile)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('trainfile', type=str)
    parser.add_argument('param', type=str)
    parser.add_argument('weightfile', type=str)
    parser.set_defaults(func=neural_network)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    args.func(args)



if __name__=='__main__':
    main()
