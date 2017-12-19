import numpy as np
import loss_functions as loss
import copy
from matplotlib import pyplot as plt

def stochastic_gradient_descent(neural_network, X_train, y_train, X_val, y_val, eta, cycles, momentum_rate=0, verbose=False):

    np.random.seed(1)
    best_weights = []
    best_biases = []
    neural_network.best_val_bin_err = np.inf

    # For plotting
    neural_network.training_history = []
    neural_network.validation_history = []
    neural_network.training_history_bin_err = []
    neural_network.validation_history_bin_err = []

    validation_indices = np.arange(0, len(X_val))
    training_indices = np.arange(0, len(X_train))


    for i in np.arange(cycles):  # Number of epochs/cycles through given data
        print('Epoch: {0}'.format(i))

        # INIT HISTORY LISTS
        current_cycle_loss_values = []  # Collect loss values for each cycle
        current_cycle_validation_values = []
        current_cycle_train_bin_errs = []
        current_cycle_val_bin_errs = []

        # SHUFFLE TRAININD DATA
        np.random.shuffle(training_indices)
        np.random.shuffle(validation_indices)

        # INIT MATRICES FOR MOMENTUM OR ADAGRAD GD
        if momentum_rate == 0:
            cache, cache_biases = adagrad_init(neural_network)
        else:
            deltaW, deltaW_biases = momentum_init(neural_network)

        for training_idx in training_indices:

            # GET SINGLE TRAINING DATA POINT
            training_example = np.array(X_train[training_idx])
            training_label = np.array(y_train[training_idx])

            # FORWARD PASS
            current_pred = neural_network.forward_pass(training_example)

            # LOSS AND ERROR EVALUATION FOR A DATA POINT IN BATCH W.R.T. OLD WEIGHTS
            current_train_loss_value = loss.binary_cross_entropy(neural_network, current_pred, training_label)
            current_train_bin_error = loss.hamming_distance(training_label, [1 if x > 0.5 else 0 for x in current_pred])
            current_cycle_loss_values.append(current_train_loss_value)
            current_cycle_train_bin_errs.append(current_train_bin_error)

            # BACKWARD PASS
            grad, grad_biases = loss.binary_cross_entropy_grad(neural_network, current_pred, training_label)

            # ==================================================================
            # UPDATE WEIGHTS
            # ==================================================================
            for layer in neural_network.weights.keys():
                if momentum_rate == 0:
                    cache[layer] += grad[layer]**2
                    cache_biases[layer] += grad_biases[layer]**2
                    neural_network.weights[layer] += eta * grad[layer] / (np.sqrt(cache[layer] + 1e-8))
                    neural_network.biases[layer] += eta * grad_biases[layer] / (np.sqrt(cache_biases[layer] + 1e-8))
                else:
                    deltaW[layer] = momentum_rate * deltaW[layer] + eta * grad[layer]
                    deltaW_biases[layer] = momentum_rate * deltaW_biases[layer] + eta * grad_biases[layer]
                    neural_network.weights[layer] += deltaW[layer]
                    neural_network.biases[layer] += deltaW_biases[layer]

            # ==================================================================

        # VALIDATION
        for validation_idx in validation_indices:
            validation_example = np.array(X_val[validation_idx])
            validation_label = np.array(y_val[validation_idx])

            validation_pred = neural_network.forward_pass(validation_example)
            current_val_value = loss.binary_cross_entropy(neural_network, validation_pred, validation_label)
            current_val_bin_err = loss.hamming_distance(validation_label, [1 if x > 0.5 else 0 for x in validation_pred])


            current_cycle_validation_values.append(current_val_value)
            current_cycle_val_bin_errs.append(current_val_bin_err)

        # RECORD HISTORY
        neural_network.training_history.append(np.average(current_cycle_loss_values))

        average_train_bin_err = np.average(current_cycle_train_bin_errs)
        neural_network.training_history_bin_err.append(average_train_bin_err)
        neural_network.validation_history.append(np.average(current_cycle_validation_values))

        average_val_bin_err = np.average(current_cycle_val_bin_errs)
        neural_network.validation_history_bin_err.append(average_val_bin_err)

        # SAVE BEST WEIGHTS FOUND SO FAR W.R.T. LOWEST VALIDATION BINARY ERROR
        if average_val_bin_err < neural_network.best_val_bin_err:
            neural_network.best_val_bin_err = average_val_bin_err
            best_weights = copy.deepcopy(neural_network.weights)
            best_biases = copy.deepcopy(neural_network.biases)


        # INFO DURING TRAINING
        if verbose == True:
            print('Epoch: {0}, Avg Bin Err (Train): {1}, Avg Bin Err (Val) {2}'.format(i, average_train_bin_err, average_val_bin_err))

        # graph(neural_network, eta, momentum_rate)

    # 'RETURN' BEST NN
    neural_network.weights = best_weights
    neural_network.biases = best_biases

def momentum_init(neural_network):
    deltaW = {layer : np.zeros((neural_network.num_neurons_all_layers[layer], neural_network.num_neurons_all_layers[layer - 1])) \
            for layer in np.arange(1, len(neural_network.num_neurons_all_layers))}
    deltaW_biases = {layer : np.zeros((neural_network.num_neurons_all_layers[layer], )) \
            for layer in np.arange(1, len(neural_network.num_neurons_all_layers))}

    return deltaW, deltaW_biases


def adagrad_init(neural_network):
    cache = {layer : np.zeros((neural_network.num_neurons_all_layers[layer], neural_network.num_neurons_all_layers[layer - 1])) \
            for layer in np.arange(1, len(neural_network.num_neurons_all_layers))}
    cache_biases = {layer : np.zeros((neural_network.num_neurons_all_layers[layer], )) \
            for layer in np.arange(1, len(neural_network.num_neurons_all_layers))}

    return cache, cache_biases
