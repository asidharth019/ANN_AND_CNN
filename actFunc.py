import numpy as np


def actFunc(act_func_type, pre_act):
    # This function is used for calculating the activation
    # using sigmoid or tanh function

    h = np.zeros((pre_act.shape[0], 1))
    if act_func_type == "sigmoid":
            h = 1.0 / (1.0 + np.exp(np.multiply(-1, pre_act)))
    elif act_func_type == "tanh":
            h = (np.exp(pre_act) - np.exp(np.multiply(-1, pre_act))) / (np.exp(pre_act) + np.exp(np.multiply(-1, pre_act)))
    return h