import numpy as np
import argparse
import pandas as pd
import randomInitializer as randInit
import optimizeFuncLoop as ofl
import optimizeFuncNoLoop as ofnl
import matplotlib.pyplot as plt
import actFunc as af
import outputFunc as of
import normaliseData as nod
import nag
import adam
import gradientDescent as gd
import momentum as mom
import pickle

# Mannualy Initializing the parameters
sizes = ["300", "300", "300"]
num_hidden = 3
activation = "sigmoid"
batch_size = 20
loss = "ce"
opt = "gd"
lr = 0.001
anneal = False
momentum = 0.1
save_dir = "parameters/"
expt_dir = ""
train = "train.csv"
test = "test.csv"
gamma = 0.001

# Unpacking of the data
train_data_csv = pd.read_csv(train)
test_data_csv = pd.read_csv(test)
val_data_csv = pd.read_csv("val.csv")

train_data = train_data_csv.values
test_data = test_data_csv.values
val_data = val_data_csv.values

# Further dividing the data in to label and input data
train_input = train_data[:, 1:train_data.shape[1] - 1]
train_label = train_data[:, train_data.shape[1] - 1]

val_input = val_data[:, 1:val_data.shape[1] - 1]
val_label = val_data[:, val_data.shape[1] - 1]

test_input = test_data[:, 1:]

# This part of code is for the normalization of the data
# 1. Mean Normalize,  2. Mean and Variance Normalize, 3. Max-Min
#  "feature" or "image" or "255"
type_normalize = 2
type_fea_image = "feature"
means = np.mean(train_input, axis=0).reshape(1, train_input.shape[1])
stds = np.std(train_input, axis=0).reshape(1, train_input.shape[1])
train_input= nod.normaliseData(train_input, type_normalize, type_fea_image, means, stds)
val_input = nod.normaliseData(val_input, type_normalize, type_fea_image, means, stds)
test_input = nod.normaliseData(test_input,type_normalize, type_fea_image, means, stds)

# inputs_num = len(test_data[1, 1:])
inputs_num = train_input.shape[1]
outputs_num = 10
epochs = 20
layers = num_hidden+1

# To make the string of hidden neurons into array
hidden_layer_sizes = [int(i) for i in sizes]

# Initializing of the Weights and Biases
weights = [None] * layers
biases = [None] * layers

weights, biases = randInit.randomInitializer(weights, biases, inputs_num, outputs_num, layers,
                                             hidden_layer_sizes)


if opt == "gd":
    total_loss_train, total_loss_val, train_weights, train_biases = gd.gradientDescent(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, gamma, expt_dir)
elif opt == "momentum":
    total_loss_train, total_loss_val, train_weights, train_biases = mom.momentum(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, gamma, expt_dir)
elif opt == "nag":
    total_loss_train, total_loss_val, train_weights, train_biases = nag.nag(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, gamma, expt_dir)
elif opt == "adam":
    total_loss_train, total_loss_val, train_weights, train_biases = gd.gradientDescent(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, gamma, expt_dir)
print(total_loss_train)

total_loss_train = total_loss_train.reshape(total_loss_train.shape[0], 1)
total_loss_val = total_loss_val.reshape(total_loss_val.shape[0], 1)
total_loss_val = total_loss_val*(train_input.shape[0]/val_input.shape[0])
epoch_count = np.arange(1, epochs+1)
epoch_count = epoch_count.reshape(epoch_count.shape[0], 1)

plt.figure(1)
plt.plot(epoch_count, total_loss_train,'r--', label='train')
plt.plot(epoch_count,total_loss_val, 'b--', label='validation')
plt.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=0.)
plt.xlabel('epochs')
plt.ylabel('Outputloss')
plt.title('The plot for training and validation loss')
plt.show()

# This part is for the Validation of the data
trail_ones = np.ones((1, val_input.shape[0]))
modified_val_data = np.append(val_input.T, trail_ones, axis=0)
pre_act = [0]
for i in range(0, layers):
    modified_w_b = np.append(train_weights[i].T, train_biases[i], axis=1)
    pre_act = np.matmul(modified_w_b, modified_val_data)
    act = af.actFunc(activation, pre_act)
    modified_val_data = np.append(act, np.ones((1, act.shape[1])), axis=0)

pred_val_outputs = of.outputFunc(pre_act)
pred_val_labels = np.argmax(pred_val_outputs, axis=0)

pred_correct = np.sum(pred_val_labels == val_label.T)

actual_count = val_label.shape[0]
print(" The accuracy on the validation data is ", (pred_correct))
print(" The accuracy on the validation data is ", (actual_count))
print(" The accuracy on the validation data is ", (pred_correct/actual_count))


# For making the file on putting the weights and biases
filename = save_dir + "Parameters_neurons"+str(sizes[0])+"_ecpochs"+str(epochs)+"_"+opt+"_lr"+str(lr)+"_act"+activation+"_layer"+str(num_hidden)+"_loss"+loss+"_.pickle"
with open(filename, "wb") as output_file:
    pickle.dump([train_weights, train_biases], output_file)


# For storing the test sample
trail_ones = np.ones((1, test_input.shape[0]))
modified_test_data = np.append(test_input.T, trail_ones, axis=0)
pre_act = [0]
for i in range(0, layers):
    modified_w_b = np.append(train_weights[i].T, train_biases[i], axis=1)
    pre_act = np.matmul(modified_w_b, modified_test_data)
    act = af.actFunc(activation, pre_act)
    modified_test_data = np.append(act, np.ones((1, act.shape[1])), axis=0)

pred_test_outputs = of.outputFunc(pre_act)
pred_test_labels = np.argmax(pred_test_outputs, axis=0)
ids = np.arange(test_input.shape[0])

headers = ["id", "label"]
test_filename = "sample_sub.csv"
df = pd.DataFrame([ids, pred_test_labels])
df = df.transpose()
df.to_csv("test_sample/"+test_filename, index=False, header=headers)
