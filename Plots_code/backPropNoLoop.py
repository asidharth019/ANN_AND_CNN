import numpy as np
import diffLoss as dl


def backPropNoLoop(pre_act, act, predicted_output, actual_output, loss, layers, weights, input_data, act_func_type):
    # In this we will apply the back propagation on each
    # input one by one

    dw = [None] * layers
    db = [None] * layers
    if loss == "sq":
        temp1 = np.multiply(actual_output,predicted_output)
        temp2 = np.sum(temp1,axis=0).reshape(1, actual_output.shape[1])
        dal = 2*np.multiply((predicted_output - actual_output),(temp1 - np.multiply(predicted_output,temp2)))
        for i in range(layers - 1, 0, -1):
            # Compute gradients w.r.t parameters
            dw[i] = np.matmul(dal, act[i - 1].T).T
            db[i] = (dal.sum(axis=1) / (dal.shape[1])).reshape(dal.shape[0], 1)
            # Compute gradient w.r.t layer below
            dh = np.matmul(weights[i], dal)
            # Compute gradients w.r.t layer below(pre-activation)
            dal = np.multiply(dh, dl.diffLoss(act_func_type, pre_act[i - 1]))

        # Compute the gradient w.r.t parameters for first layer with h as input
        dw[0] = np.matmul(dal, input_data).T
        db[0] = (dal.sum(axis=1) / (dal.shape[1])).reshape(dal.shape[0], 1)
    elif loss == "ce":
        dal = predicted_output - actual_output
        for i in range(layers - 1, 0, -1):
            # Compute gradients w.r.t parameters
            dw[i] = np.matmul(dal, act[i - 1].T).T
            db[i] = (dal.sum(axis=1)/(dal.shape[1])).reshape(dal.shape[0], 1)
            # Compute gradient w.r.t layer below
            dh = np.matmul(weights[i], dal)
            # Compute gradients w.r.t layer below(pre-activation)
            dal = np.multiply(dh, dl.diffLoss(act_func_type, pre_act[i-1]))

        # Compute the gradient w.r.t parameters for first layer with h as input
        dw[0] = np.matmul(dal, input_data).T
        db[0] = (dal.sum(axis=1)/(dal.shape[1])).reshape(dal.shape[0], 1)


    return dw, db