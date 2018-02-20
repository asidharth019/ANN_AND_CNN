import numpy as np
import actFunc as af
import outputFunc as of
import calLoss as cl
import backProp as  bp


def optimizeFuncLoop(data, labels, layers, bias, weight, act_func_type, batch, loss, epochs, lr):
    # In this we are writing the gradient descent

    total_loss = np.zeros(epochs)
    for e in range(0, epochs):
        dw = [0] * layers
        db = [0] * layers
        itr = 0
        while itr < data.shape[0]:
            batch_count = 0
            while batch_count < batch and itr < data.shape[0]:
                    # Calculation of a_i and h_i
                    a = [None] * layers
                    h = [None] * layers
                    temp = data[itr, :].T
                    # Calculating the a and h for the hidden layers
                    for k in range(0, layers - 1):
                        x_part = (np.matmul(weight[k].T, temp)).reshape((np.matmul(weight[k].T, temp)).shape[0], 1)
                        a[k] = (x_part + bias[k])
                        h[k] = af.actFunc(act_func_type, a[k])
                        temp = h[k]

                    # Calculating the a and h for the output layer
                    x_part = (np.matmul(weight[layers-1].T, temp)).reshape((np.matmul(weight[layers-1].T, temp)).shape[0], 1)
                    a[layers - 1] = (bias[layers - 1] + x_part)
                    h[layers - 1] = of.outputFunc(a[layers - 1])
                    predicted_output = h[layers - 1]

                    # Calculating the actual output through the label
                    actual_output = np.zeros(predicted_output.shape[0])
                    actual_output = actual_output.reshape(actual_output.shape[0], 1)
                    actual_output[labels[itr]] = 1

                    total_loss[e] = total_loss[e] + cl.calLoss(predicted_output, actual_output, act_func_type, labels[itr],
                                                               loss)

                    # Back-Propagation step for each element
                    grad_w, grad_b = bp.backProp(a, h, predicted_output, actual_output, loss, layers, weight, data[itr, :], act_func_type)

                    # Accumulating the change of parameters
                    for i in range(0, layers):
                        dw[i] = dw[i] + grad_w[i]
                        db[i] = db[i] + grad_b[i]

                    itr = itr + 1
                    batch_count += 1


            # Updating the weight to the new weight using the  dw and db
            for i in range(0, layers):
                weight[i] = weight[i] - lr * dw[i]
                bias[i] = bias[i] - lr * db[i]
            itr = itr + 1

    return total_loss, weight, bias