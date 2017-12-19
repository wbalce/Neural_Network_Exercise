import numpy as np
from random import *
from data_sets import *
import neural_network as nn
from matplotlib import pyplot as plt
import loss_functions as loss

# ==============================================================================
# TRANING DATA SETTINGS
# ==============================================================================
batch_size = 1 # Size of training batch (positive int or 'all')
num_of_examples = 2000  # Max number of examples for each label


# ==============================================================================
# HYPERPARAMETER SETTINGS
# ==============================================================================
size_input = 5  # The size of v+ and v-
neurons_dummy = [1]
neurons_hidden_layers_1 = [50, 30]
neurons_hidden_layers_2 = [40, 20, 10]
# neurons_hidden_layers_3 = [40, 10]
# neurons_hidden_layers_4 = [40, 30]

model_arch = [neurons_dummy, neurons_hidden_layers_1, neurons_hidden_layers_2]
# model_arch = [neurons_dummy, neurons_hidden_layers_1, neurons_hidden_layers_2, neurons_hidden_layers_3, neurons_hidden_layers_4]
# ==============================================================================
# OPTIMIZATION SETTINGS
# ==============================================================================
learning_rate_1 = 0.001
learning_rate_2 = 0.01
# learning_rate_3 = 0.1
model_lr = [learning_rate_1, learning_rate_2]

# momentum_rate_1 = 0.5
# momentum_rate_2 = 0.6
# momentum_rate_3 = 0.7
# model_mr = [momentum_rate_1, momentum_rate_2]
# model_mr = [0]

number_of_epochs = 10000


# ==============================================================================
# GET DATA SETS (TRAINING AND TESTING SETS)
# ==============================================================================
labels, training_data, validation_data, testing_data = generate_data_sets(size_input, num_of_examples)

y_train = []
X_train = []
y_test = []
X_test = []
X_val = []
y_val = []
for label in labels.keys():
    y_train = y_train + [labels[label]]*len(training_data[label])
    X_train = X_train + training_data[label]

    if label in testing_data:
        y_test = y_test + [labels[label]]*len(testing_data[label])
        X_test = X_test + testing_data[label]

    if label in validation_data:
        y_val = y_val + [labels[label]]*len(validation_data[label])
        X_val = X_val + validation_data[label]


# ==============================================================================
# NEURAL NETWORK TRAINING
# ==============================================================================

settings = np.stack(np.meshgrid(model_arch, model_lr), -1).reshape(-1, 2)  # No MR

dummy_rows = []
for i in np.arange(0, len(settings)):
    if len(settings[i,0]) == 1:
        dummy_rows.append(i)
settings = np.delete(settings, dummy_rows, axis=0)
print(settings)

models = []
for i in np.arange(0, len(settings)):
    hn = settings[i, 0]
    lr = settings[i, 1]
    # mr = settings[i, 2]

    mr = 'NA'
    print(hn)
    models.append(nn.Neural_Network(size_input, hn))
    models[i].train(X_train, y_train, X_val, y_val, lr, number_of_epochs, batch_size, mr)
    plt.figure(i)
    plt.subplot(211)
    plt.title("HN: {0}, LR: {1}, MR: {2}".format(hn, lr, mr))
    # plt.xlabel('Epoch')
    plt.ylabel('Average Loss Value')
    plt.plot(models[i].training_history, 'k-', models[i].validation_history, 'r-')
    plt.subplot(212)
    plt.plot(models[i].training_history_bin_err, 'b-', models[i].validation_history_bin_err, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Binary Error')
    plt.savefig('figure_' + str(i) + '.svg')

for i in np.arange(0, len(settings)):
    hn = settings[i, 0]
    lr = settings[i, 1]
    # mr = settings[i, 2]
    mr = 'NA'
    print("HN: {0}, LR: {1}, MR: {2}, Best Avg Val Bin Err: {3}".format(hn, lr, mr, models[i].best_val_bin_err))


for model in models:
    test_acc = []
    test_err = []
    for i in np.arange(0, len(X_test)):
        y = y_test[i]
        x = np.array(X_test[i])
        pred = model.classify(x)
        test_err.append(loss.hamming_distance(y, pred))
        test_acc.append(pred == y)
    print('HN: {0}, Avg Test Bin Err: {1}, Accuracy: {2}'.format(model.num_neurons_hidden_layers, np.average(test_err), np.average(test_acc)))
