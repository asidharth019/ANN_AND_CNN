import numpy as np


def crossEntropy(a, h, layers, predOutput, actual, weight, loss):
    # In this we are calculating the loss function
    # which is the cross entropy in this case

    dw = [None] * layers
    db = [None] * layers
    daL = -(actual - predOutput)
    for i in range(layers, 0)
        dw[i] = np.matmul(daL, (h[i - 1]).T)
        db[i] = daL
        if i != 1
            dh = np.matmul((weight[i].T) * daL)
            daL = np.multiply(dh, diffLoss(a[i - 1], loss))

    return dw, db