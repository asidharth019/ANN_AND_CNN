import numpy as np


def outputFunc(preAct):
    # It is the output function in which we will apply
    # the SOFTMAX function as we require the classification
    # in this task
    expon = np.exp(preAct)
    exp_sum = np.sum(expon, axis=0)
    h = expon/exp_sum

    return h