import numpy as np

def hamming_distance(bin_list_1, bin_list_2):
    return np.sum([1 if bin_list_1[x] != bin_list_2[x] else 0 for x in np.arange(0, len(bin_list_1))])


def binary_cross_entropy(neural_network, nn_output, y_train):
    """
    NOTE:   SOMETIMES RECEIVES ERROR

    Input:  1. The NN output for a training data example (array)
            2. Output of neural network on training data exampled (array)
            3. Training data label (list)

    Output: 1. Cross entropy loss for multi-label classification (float)
    """
    return -np.average(y_train*np.log(nn_output) + (1-y_train)*np.log(1-nn_output))


def binary_cross_entropy_grad(neural_network, pred, y_train):
    """
    Input:  1. The neural network to train (object)
            2. The NN output for a training data example (array)
            3. The training data label (array)

    Output: 1. Gradient with respect to each weight matrix (dictionary)
    """
    # Construct partial deriv of loss w.r.t. its input
    numer = y_train * (1 - pred) - pred * (1 - y_train)
    denom = pred * (1 - pred)
    J = numer/denom/neural_network.num_bits

    grad = {layer : 0 for layer in np.arange(1, len(neural_network.weights))}
    grad_biases = {layer : 0 for layer in np.arange(1, len(neural_network.biases))}
    for layer in reversed(np.arange(1, len(neural_network.num_neurons_all_layers))):

        l_to_update_output = neural_network.output_per_layer[layer]
        l_to_update_output_size = neural_network.weights[layer].shape[0]
        l_to_update_input = neural_network.output_per_layer[layer - 1]
        l_to_update_input_size = neural_network.weights[layer].shape[1]

        # Construct partial deriv of loss w.r.t. weights of the layer we want to update
        Partial = []
        for neuron in np.arange(0, l_to_update_output_size):
            l_to_update_partial = np.zeros((l_to_update_output_size, l_to_update_input_size))
            l_to_update_partial[neuron, :] = l_to_update_output[neuron] * l_to_update_input
            partial_i = np.dot(J, l_to_update_partial)
            Partial.append(partial_i)
        grad[layer] = np.array(Partial)

        # Construct partial deriv of loss w.r.t. biases of the layer we want to update
        B = l_to_update_output * (1 - l_to_update_output)
        partial_biases = np.diag(B.tolist())
        grad_biases[layer] = np.dot(J, partial_biases)

        # Update J
        W = neural_network.weights[layer]
        C = np.tile(B, (l_to_update_input_size, 1)).T
        J = np.dot(J, C*W)

    return grad, grad_biases
