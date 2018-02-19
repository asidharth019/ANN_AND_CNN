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
epochs = 3

sizes = np.array([[50],[100],[200],[300]])
num_hidden = 1
layers = num_hidden+1

activation = "sigmoid"
batch_size = 5000
loss = "ce"
opt = "adam"
lr = 0.001
anneal = False
momentum = 0.1
total_loss_train, total_loss_val = [0]*4, [0]*4


for i in range(0,4):
        weights = [0] * layers
        biases = [0] * layers
        weights, biases = randInit.randomInitializer(weights, biases, inputs_num, outputs_num, layers,
                                                     sizes[i])
        total_loss_train[i], total_loss_val[i], train_weights, train_biases = adam.adam(train_input, train_label,
                                                                                               val_input, val_label, layers,
                                                                                               biases, weights, activation,
                                                                                               batch_size,
                                                                                               loss,
                                                                                               epochs,
                                                                                                  lr, anneal, gamma, expt_dir)
        print("value of i is %d",i)

print(total_loss_train,total_loss_val)

# for i in range(0,4):
#     total_loss_train[i] = total_loss_train[i].reshape(total_loss_train[i].shape[0], 1)
#     total_loss_val[i] = total_loss_val[i].reshape(total_loss_val[i].shape[0], 1)


epoch_count = np.arange(1, epochs+1)
epoch_count = epoch_count.reshape(epoch_count.shape[0], 1)

plt.figure(1)
plt.plot(epoch_count, total_loss_train[0].T,'r-', label='50')
plt.plot(epoch_count, total_loss_train[1].T,'b-', label='100')
plt.plot(epoch_count, total_loss_train[2].T,'g-', label='200')
plt.plot(epoch_count, total_loss_train[3].T,'y-', label='300')
plt.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=0.)
plt.xlabel('epochs')
plt.ylabel('Training Loss')
plt.title('The Plot for training loss with different number of hidden neurons')
# plt.savefig("figures/Train_"+str(num_hidden)+".jpg",dpi=72)
plt.show()


plt.figure(2)
plt.plot(epoch_count, total_loss_val[0].T,'r-', label='50')
plt.plot(epoch_count, total_loss_val[1].T,'b-', label='100')
plt.plot(epoch_count, total_loss_val[2].T,'g-', label='200')
plt.plot(epoch_count, total_loss_val[3].T,'y-', label='300')
plt.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=0.)
plt.xlabel('epochs')
plt.ylabel('Validation Loss')
plt.title('The Plot for validation loss with different number of hidden neurons')
# plt.savefig("figures/Validation_"+str(num_hidden)+".jpg",dpi=72)
plt.show()
