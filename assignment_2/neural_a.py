import numpy as np
import argparse
import sys
import math



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, hidden_layers, feature_size, output_size):
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
        
        i = len(self.weights) - 1
        for weight, bias, activ in (zip(reversed(self.weights), reversed(self.biases), reversed(activs[:-1]))):
            dW = activ.T @ delta
            dB = np.sum(delta, axis=0, keepdims=True)
            delta = (delta @ weight.T) * (1 - activ) * activ
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
    with open(args.trainfile) as f:
        train = np.loadtxt(f, delimiter=',')
    x = train[:, :-1]
    y = train[:, -1:]

    with open(args.param) as f:
        param = f.readlines()
    strategy = int(param[0])
    base_lr = float(param[1])
    iterations = int(param[2])
    batch_size = int(param[3])
    hidden_layers = [int(x) for x in param[4].split()]

    batches = x.shape[0] // batch_size
    nn = NeuralNetwork(hidden_layers=hidden_layers, feature_size=x.shape[1], output_size=1)

    i = 0
    for iter in range(iterations):
        if strategy == 1:
            lr = base_lr
        else:
            lr = base_lr / math.sqrt(iter + 1)
        
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
