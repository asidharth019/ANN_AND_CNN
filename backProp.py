import numpy as np
import diffLoss as dl


def backProp(pre_act, act, predicted_output, actual_output, loss, layers, weights, input_data, act_func_type):
    # In this we will apply the back propagation on each
    # input one by one

    dw = [None] * layers
    db = [None] * layers
    if loss == "sq":
        c = 1
    elif loss == "ce":
        predicted_output = predicted_output.reshape(predicted_output.shape[0], 1)
        dal = predicted_output - actual_output
        for i in range(layers - 1, 0, -1):
            # Compute gradients w.r.t parameters
            dw[i] = np.matmul(dal, act[i - 1].T).T
            db[i] = dal
            # Compute gradient w.r.t layer below
            dh = np.matmul(weights[i], dal)
            # Compute gradients w.r.t layer below(pre-activation)
            dal = np.multiply(dh, dl.diffLoss(act_func_type, pre_act[i-1]))

        # Compute the gradient w.r.t parameters for first layer with h as input
        input_data = input_data.reshape(input_data.shape[0], 1)
        dw[0] = np.matmul(dal, input_data.T).T
        db[0] = dal

    return dw, db