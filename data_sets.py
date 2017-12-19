import numpy as np
from random import *

def generate_data_sets(size_input, K):
    """
    Generate all possible labels and a number of examples for each.
    Note: Inputs and weights have extra component for bias.

    Input:  1. size of v+ or v- (int)
            2. max number of examples desired for each label (int)

    Output: 1. binary labels, keys are integer labels  (dictionary, int : 1D list)
            2. training examples, keys are integer labels (dictionary, int : ND list)
            3. testing examples, keys are intger labels (dictionary, int : ND list)
    """
    np.random.seed(4)


    # GENERATE LABELS
    labels_int = np.arange(-size_input, size_input + 1)
    num_bits = len(bin(size_input)[2:]) + 1  #Extra bit required for the sign
    format_str = '{0:0' + str(num_bits) + 'b}'
    labels = {key: bin(key % (1<<num_bits))[2:] if key < 0 else format_str.format(key) for key in labels_int}
    for key in labels.keys():
        labels[key] = [int(n) for n in labels[key]]


    # GENERATE K EXAMPLES FOR EACH LABEL
    data_dict = {key : [] for key in labels.keys()}
    randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]

    part_a = np.ones(size_input).tolist()
    part_b = np.zeros(size_input).tolist()
    data_dict[size_input].append(part_a + part_b)
    data_dict[-size_input].append(part_b + part_a)

    k = 0
    while k < 25000:
        ones = np.ones(size_input).tolist()
        num_to_delete = np.random.choice(size_input)
        idx_to_delete = np.random.choice(size_input, num_to_delete, replace=False)

        zeros = np.zeros(size_input).tolist()
        num_to_insert = np.random.choice(size_input)
        idx_to_insert = np.random.choice(size_input, num_to_insert, replace=False)

        for i in range(0,size_input+1):
            if i in idx_to_delete:
                ones[i] = 0
            if i in idx_to_insert:
                zeros[i] = 1

        part_a = ones
        part_b = zeros
        candidate_label_6 = sum(part_a) - sum(part_b)
        candidate_list_6 = part_a + part_b

        # K + 1 since there's a dummy row
        if len(data_dict[candidate_label_6]) < K and not(candidate_list_6 in data_dict[candidate_label_6]):
            data_dict[candidate_label_6].append(candidate_list_6)

        # ATTEMPT TO INCREASE EXAMPLE DIVERSITY (MAINLY FOR LARGER INPUT SIZES)
        # i.e. not just reflect the the previous candidate (might be limiting for larger input sizes)
        ones_2 = np.ones(size_input).tolist()
        num_to_delete_2 = np.random.choice(size_input)
        idx_to_delete_2 = np.random.choice(size_input, num_to_delete_2, replace=False)

        zeros_2 = np.zeros(size_input).tolist()
        num_to_insert_2 = np.random.choice(size_input)
        idx_to_insert_2 = np.random.choice(size_input, num_to_insert_2, replace=False)

        for i in range(0,size_input+1):
            if i in idx_to_delete_2:
                ones_2[i] = 0
            if i in idx_to_insert_2:
                zeros_2[i] = 1

        part_c = ones_2
        part_d = zeros_2
        candidate_label_2 = sum(part_d) - sum(part_c)
        candidate_list_2 = part_d + part_c

        if len(data_dict[candidate_label_2]) < K and not(candidate_list_2 in data_dict[candidate_label_2]):
            data_dict[candidate_label_2].append(candidate_list_2)

        k = k + 1

    # CHECK: NUMBER OF EXAMPLES FOR EACH LABEL
    for key in data_dict.keys():
        print('Label: {0}, Size: {1}'.format(key, len(data_dict[key])))

    testing_data = {}
    for key in data_dict.keys():
        if len(data_dict[key]) > 1:
            test_example_idx = np.random.choice(len(data_dict[key]))
            testing_data.update({key : [data_dict[key].pop(test_example_idx)]})

    validation_data = {}
    for key in data_dict.keys():
        if len(data_dict[key]) > 1:
            val_example_idx = np.random.choice(len(data_dict[key]))
            validation_data.update({key : [data_dict[key].pop(val_example_idx)]})

    return labels, data_dict, validation_data, testing_data
