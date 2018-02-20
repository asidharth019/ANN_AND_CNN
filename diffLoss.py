import numpy as np
import actFunc as af


def diffLoss(act_func_type, pre_act):
    # In this function we will calculate the
    # differentiation of the loss functions

    b = af.actFunc(act_func_type, pre_act)
    if act_func_type == "sigmoid":
            diff = np.multiply(b, (1.0 - b))
    elif act_func_type == "tanh":
            diff = 1.0 - np.multiply(b, b)

    return diff