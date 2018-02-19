import numpy as np


def randomInitializer(weights, biases, inputs_num, outputs_num, layers, sizes):
    # In this function we have done the random initialization of the weight and
    # the biases

    itr = 0
    np.random.seed(1234)
    scaler = 0.01
    # Parameter of Input Layer
    weights[0] = scaler*np.random.rand(inputs_num, sizes[0])
    biases[0] = scaler*np.random.rand(sizes[0], 1)
    itr = itr + 1
    # Parameters of Hidden Layer
    while itr < layers-1:
        weights[itr] = scaler*np.random.rand(sizes[itr-1], sizes[itr])
        biases[itr] = scaler*np.random.rand(sizes[itr], 1)
        itr = itr + 1

    # Parameter of Output Layer
    weights[itr] = scaler*np.random.rand(sizes[itr-1], outputs_num)
    biases[itr] = scaler*np.random.rand(outputs_num, 1)

    return weights, biases
