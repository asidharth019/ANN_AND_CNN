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

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float)
parser.add_argument("--momentum", help="parameter for using the histrory", type=float)
parser.add_argument("--num_hidden", help="number of hidden layers", type=int)
parser.add_argument("--sizes", help="string of number with number of neurons in each hidden layer", type=str)
parser.add_argument("--activation", help="activation function", type=str)
parser.add_argument("--loss", help="loss function such as squared error(sq) and cross entropy(ce)", type=str)
parser.add_argument("--opt", help="Optimazation functions such as adam, gd, momentum", type=str)
parser.add_argument("--batch_size", help="Size of the batch", type=int)
parser.add_argument("--anneal", help="If true then half the learning rate", type=bool)
parser.add_argument("--save_dir", help="Directory in which the file of weights and biases will be saved", type=str)
parser.add_argument("--expt_dir", help="Diectory in which log files wil be saved", type=str)
parser.add_argument("--train", help="The training data file name", type=str)
parser.add_argument("--test", help="The testing data file name", type=str)
parser.add_argument("--val", help="The validation data file name", type=str)

arg = parser.parse_args()

# Mannualy Initializing the parameters
hidden_layer_sizes = [int(i) for i in arg.sizes.split(',')]
num_hidden = arg.num_hidden
activation = arg.activation
batch_size = arg.batch_size
loss = arg.loss
opt = arg.opt
lr = arg.lr
anneal = arg.anneal
momentum = arg.momentum
save_dir = arg.save_dir
expt_dir = arg.expt_dir
train = arg.train
test = arg.test
val = arg.val

# Unpacking of the data
train_data_csv = pd.read_csv(train)
test_data_csv = pd.read_csv(test)
val_data_csv = pd.read_csv(val)

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
train_input = nod.normaliseData(train_input, type_normalize, type_fea_image, means, stds)
val_input = nod.normaliseData(val_input, type_normalize, type_fea_image, means, stds)
test_input = nod.normaliseData(test_input,type_normalize, type_fea_image, means, stds)

inputs_num = train_input.shape[1]
outputs_num = 10
epochs = 15
layers = num_hidden+1

# Initializing of the Weights and Biases
weights = [0] * layers
biases = [0] * layers

weights, biases = randInit.randomInitializer(weights, biases, inputs_num, outputs_num, layers,
                                             hidden_layer_sizes)


if opt == "gd":
    total_loss_train, total_loss_val, train_weights, train_biases = gd.gradientDescent(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, momentum, expt_dir)
elif opt == "momentum":
    total_loss_train, total_loss_val, train_weights, train_biases = mom.momentum(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, momentum, expt_dir)
elif opt == "nag":
    total_loss_train, total_loss_val, train_weights, train_biases = nag.nag(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, momentum, expt_dir)
elif opt == "adam":
    total_loss_train, total_loss_val, train_weights, train_biases = adam.adam(train_input, train_label,
                                                                                       val_input, val_label, layers,
                                                                                       biases, weights, activation,
                                                                                       batch_size,
                                                                                       loss,
                                                                                       epochs,
                                                                                       lr, anneal, momentum, expt_dir)
# For making the file on putting the weights and biases
filename = save_dir + "Parameters_neurons"+str(hidden_layer_sizes[0])+"_ecpochs"+str(epochs)+"_"+opt+"_lr"+str(lr)+"_act"+activation+"_layer"+str(num_hidden)+"_loss"+loss+"_.pickle"
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
df.to_csv(expt_dir+test_filename, index=False, header=headers)
