import numpy as np
from random import *
import optimization_methods as om
from scipy.special import expit

class Neural_Network(object):
    """
    NOTE: Uses sigmoid activation function for every neuron

    INFO: Given two binary vectors of equal length, v+ and v-, this neural network attempts
    to output sum(v+) - sum(v-) in binary format.

    """


    def __init__(self, size_input, num_neurons_hidden_layers):
        """
        Input:  1. Size of v+ and v- (int)
                2. Number of neurons for each hidden layer (list)
                    e.g. [10, 5] for 10 neurons in first hidden layer, 5 for the second

        Output: N/A
        """
        self.size_input = size_input  # Size of v+ and v-
        self.num_neurons_hidden_layers = num_neurons_hidden_layers
        self.num_bits = len(bin(size_input)[2:]) + 1  # +1 for sign bit
        self.num_neurons_all_layers = [2*self.size_input] + self.num_neurons_hidden_layers + [self.num_bits]
        self.output_per_layer = {layer : 0 for layer in np.arange(0, len(self.num_neurons_all_layers))}
        print("Neural Network Dimensions: ", self.num_neurons_all_layers)


    def reset_weights(self):
        np.random.seed(1)
        self.weights = {layer : \
                        np.random.rand(self.num_neurons_all_layers[layer], self.num_neurons_all_layers[layer - 1]) \
                        for layer in np.arange(1, len(self.num_neurons_all_layers))}
        print("\n\nDimensions of Weights:")
        for key in self.weights.keys():
            print("Layer {0}: {1}".format(key, self.weights[key].shape))


    def reset_biases(self):
        np.random.seed(2)
        self.biases = {layer: \
                        np.random.rand(self.num_neurons_all_layers[layer]) \
                        for layer in np.arange(1, len(self.num_neurons_all_layers))}
        print("\n\nDimensions of Biases:")
        for key in self.biases.keys():
            print("Layer {0}: {1}".format(key, self.biases[key].shape))


    def train(self, X_train, y_train, X_val, y_val, eta, cycles, batch_size, momentum_rate=0):
        """
        Input:  1. Training data (array)
                2. Training labels (list)
                3. Learning rate (float)
                4. Number of cycles through training data (int)

        Output: N/A
        """
        print("lr: {0}, mr: {1}".format(eta, momentum_rate))
        self.reset_weights()
        self.reset_biases()
        self.num_data = len(X_train)
        if batch_size == 1:
            om.stochastic_gradient_descent(self, X_train, y_train, X_val, y_val, eta, cycles, momentum_rate)
        else:
            om.gradient_descent(self, X_train, y_train, X_val, y_val, eta, cycles, momentum_rate, batch_size)


    def forward_pass(self, X):
        """
        Input:  1. Binary vector to be processed (array)

        Output: 1. Raw output of the neural network (array)
        """

        output = X.T  # Transposed in case we want to upgrade to larger batch training
        self.output_per_layer[0] = output
        for layer in np.arange(1, 1 + len(self.weights.keys())):
            output = self.sigmoid(np.dot(self.weights[layer], output) + self.biases[layer])
            output = np.array([0.99999999 if x == 1.0 else 0.00000001 if x == 0.0 else x for x in output])
            self.output_per_layer[layer] = output  # Output is saved in order to calculate gradient (via backprop)
        return output


    def classify(self, x):
        """
        Input:  1. Binary vector to be processed (array)

        Output: 1. Binary output of the neural network (list)
        """
        forward_pass_output = self.forward_pass(x)
        return [1 if x > 0.5 else 0 for x in forward_pass_output]


    def sigmoid(self, x):
        """
        NOTE: Uses scipy.special.expit to avoid numerical error

        Input:  1. Linear mapping (array)

        Output: 1. Sigmoid function applied elementwise (array)
        """
        # return 1/(1+np.exp(-(x)))
        return expit(x)
