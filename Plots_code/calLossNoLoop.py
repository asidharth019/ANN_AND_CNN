import numpy as np


def calLossNoLoop(predicted_output, actual_output, act_func_type, label, loss):
    # This function is to calculate the loss generated
    # when a single input is given

    output_loss = [0]
    if loss == "ce":
        temp = np.multiply(predicted_output, actual_output)
        temp = temp.sum(axis=0)
        temp = -np.log(temp)
        output_loss = temp.sum()
    elif loss == "sq":
        x = predicted_output - actual_output
        temp = np.square(x)
        output_loss = temp.sum()

    return output_loss