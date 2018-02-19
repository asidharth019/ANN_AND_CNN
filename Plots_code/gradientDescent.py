import numpy as np
import actFunc as af
import outputFunc as of
import calLossNoLoop as clnl
import backPropNoLoop as  bpnl
from sklearn.preprocessing import OneHotEncoder


def gradientDescent(input_data, input_label, val_data, val_label ,layers, bias, weight, act_func_type, batch, loss, epochs, lr, anneal, gamma, expt_dir):
    # In this we are writing the gradient descent

    input_label = input_label.reshape(input_label.shape[0], 1)
    val_label = val_label.reshape(val_label.shape[0], 1)
    total_train_loss = np.zeros(epochs)
    total_val_loss = np.zeros(epochs)
    for e in range(0, epochs):
        batch_count = 0
        while batch_count < input_data.shape[0]:
                    if (batch_count+batch) <= input_data.shape[0]:
                        data = input_data[batch_count:(batch_count+batch), :]
                        labels = input_label[batch_count:(batch_count+batch), :]
                    elif (batch_count+batch) > input_data.shape[0]:
                        data = input_data[batch_count:input_data.shape[0], :]
                        labels = input_label[batch_count:(batch_count + batch), :]
                    batch_count = batch_count + batch
                    # Calculation of a_i and h_i
                    a = [None] * layers
                    h = [None] * layers
                    # Calculating the a and h
                    trail_ones = np.ones((1, data.shape[0]))
                    modified_data = np.append(data.T, trail_ones, axis=0)
                    for i in range(0, layers):
                        modified_w_b = np.append(weight[i].T, bias[i], axis=1)
                        a[i] = np.matmul(modified_w_b, modified_data)
                        h[i] = af.actFunc(act_func_type, a[i])
                        modified_data = np.append(h[i], np.ones((1, h[i].shape[1])), axis=0)

                    h[layers-1] = of.outputFunc(a[layers-1])
                    predicted_output = h[layers - 1]

                    # Calculating the actual output through the label
                    enc = OneHotEncoder(n_values=predicted_output.shape[0])
                    actual_output = enc.fit_transform(labels).toarray().T

                    # Back-Propagation step for each element
                    dw, db = bpnl.backPropNoLoop(a, h, predicted_output, actual_output, loss, layers, weight, data, act_func_type)
                    # Updating the weight to the new weight using the  dw and db
                    for i in range(0, layers):
                            weight[i] = weight[i] - lr * dw[i]
                            bias[i] = bias[i] - lr * db[i]

                    total_train_loss[e] = total_train_loss[e] + clnl.calLossNoLoop(predicted_output, actual_output,
                                                                                   act_func_type, labels, loss)

        # For the loss at the validation
        a = [None] * layers
        h = [None] * layers
        trail_ones = np.ones((1, val_data.shape[0]))
        modified_data = np.append(val_data.T, trail_ones, axis=0)
        for i in range(0, layers):
            modified_w_b = np.append(weight[i].T, bias[i], axis=1)
            a[i] = np.matmul(modified_w_b, modified_data)
            h[i] = af.actFunc(act_func_type, a[i])
            modified_data = np.append(h[i], np.ones((1, h[i].shape[1])), axis=0)

        h[layers - 1] = of.outputFunc(a[layers - 1])
        predicted_val_output = h[layers - 1]

        # Calculating the actual output through the label
        enc = OneHotEncoder(n_values=predicted_val_output.shape[0])
        actual_val_output = enc.fit_transform(val_label).toarray().T

        total_val_loss[e] = clnl.calLossNoLoop(predicted_val_output, actual_val_output, act_func_type, val_label, loss)

        if anneal == True and (e+1)%5 == 0:
            lr = lr / 2
        print("epoch:", e)
    return total_train_loss, total_val_loss, weight, bias
