import numpy as np


def calLoss(predicted_output, actual_output, act_func_type, label, loss):
    # This function is to calculate the loss generated
    # when a single input is given

    output_loss = [0]
    if loss == "ce":
        output_loss = -np.log(predicted_output[label])
    elif loss == "sq":
        x = predicted_output - actual_output
        output_loss = sum(np.multiply(x, x).T / (predicted_output.shape[0]))

    return output_loss