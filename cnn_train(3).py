import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

train_file_path = '../data/train.csv'
valid_file_path = '../data/val.csv'
test_file_path = '../data/test.csv'

# Training Parameters
learning_rate = 0.1
num_steps = 5
batch_size = 200
batch_start = 0
batch_end = batch_start + batch_size
display_step = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout_hl = 0.5  # Dropout, probability to keep hidden layer of fully connected units
dropout_il = 0.8  # Dropout, probability to keep first fully connected layer of cnn
init_mode = 1  # 1 for Xavier, 2 for He

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
cnn_mode = tf.placeholder(tf.string)  # specify training/validation/testing mode


# load training, valid, test data
def load_data(file_path, data_kind):
    # data_kind specify train/valid/test data
    print('loading ' + data_kind + ' data...')
    file = np.genfromtxt(file_path, dtype=float, delimiter=',')

    if data_kind == 'train':
        file = np.delete(file, 0, axis=0)  # delete first row
        file = np.delete(file, 0, axis=1)  # delete first column
        data = file[:, 0:-1]
        label = file[:, -1]
        return data, label

    elif data_kind == 'val':
        file = np.delete(file, 0, axis=0)  # delete first row
        file = np.delete(file, 0, axis=1)  # delete first column
        data = file[:, 0:-1]
        label = file[:, -1]
        return data, label

    else:  # test data
        file = np.delete(file, 0, axis=1)  # delete first column
        file = np.delete(file, 0, axis=0)  # delete first row
        data = file
        return data


# MNIST Fashion data
tf.set_random_seed(1)
train_data, train_label = load_data(file_path=train_file_path, data_kind='train')
valid_data, valid_label = load_data(file_path=valid_file_path, data_kind='val')
test_data = load_data(file_path=test_file_path, data_kind='test')


def cnn_model_fn(features):

    """Model function for CNN."""
    batch_init = tf.contrib.layers.xavier_initializer() if init_mode == 1 \
        else tf.keras.initializers.he_normal(seed=None)

    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3],
                             kernel_initializer=batch_init,
                             bias_initializer=batch_init, padding="same", activation=tf.nn.relu, name='conv1')

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="same")

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                             kernel_initializer=batch_init,
                             bias_initializer=batch_init,
                             padding="same", activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="same")

    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                             kernel_initializer=batch_init,
                             bias_initializer=batch_init,
                             padding="same", activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3],
                             kernel_initializer=batch_init,
                             bias_initializer=batch_init,
                             padding="same", activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding="same")

    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 256])

    fc1 = tf.layers.dense(inputs=pool3_flat, units=1024,
                          kernel_initializer=batch_init,
                          bias_initializer=batch_init, activation=tf.nn.relu, name='fc1')

    fc1 = tf.layers.dropout(inputs=fc1, rate=dropout_il) if cnn_mode == 'train' else fc1  # Dropout input layer

    fc2 = tf.layers.dense(inputs=fc1, units=1024, kernel_initializer=batch_init,
                          bias_initializer=batch_init, activation=tf.nn.relu)

    fc2 = tf.layers.dropout(inputs=fc2, rate=dropout_hl) if cnn_mode == 'train' else fc2  # Dropout input layer

    fc3 = tf.layers.dense(inputs=fc2, units=1024,
                          kernel_initializer=batch_init,
                          bias_initializer=batch_init, activation=tf.nn.relu)

    fc3_bn = tf.nn.batch_normalization(x=fc3, mean=0, variance=1, scale=1, offset=0,
                                       variance_epsilon=1e-6) if cnn_mode == 'train' else fc3  # Batch Normalization

    fc4 = tf.layers.dense(inputs=fc3_bn, units=10, kernel_initializer=batch_init,
                          bias_initializer=batch_init)

    return fc4


def batch_data(batch_start_index, batch_end_index, data_kind):
    #  Returns batch data for training/validation/testing
    if data_kind == "train":
        data, label = train_data[batch_start_index:batch_end_index, :], \
                      train_label[batch_start_index:batch_end_index]

        label = label.astype(np.int)
        label_one_hot = np.zeros([label.shape[0], num_classes])
        label_one_hot[np.arange(label.shape[0]), label] = 1

        return data, label_one_hot

    elif data_kind == "valid":
        data, label = valid_data[batch_start_index:batch_end_index, :], \
                      valid_label[batch_start_index:batch_end_index]

        label = label.astype(np.int)
        label_one_hot = np.zeros([label.shape[0], num_classes])
        label_one_hot[np.arange(label.shape[0]), label] = 1
        return data, label_one_hot

    elif data_kind == "test":  # test data
        data = test_data[batch_start_index:batch_end_index, :]
        return data


def plot_figure(x_axis, y_axis, x_label, y_label, title, figure_num):
    plt.figure(figure_num)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x_axis, y_axis)
    plt.show()


def heat_map(w, input_channel=0):  # w is 3D matrix of size 3*3*1*64

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = 64
    num_grids = 8

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Model output
logits = cnn_model_fn(X)

# Define loss and optimizer
loss_op = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Test Prediction
test_pred_tensor = tf.argmax(logits, 1)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Fully connected layer weight and biased
conv1_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1')[0]

# Compute Training, Validation, Testing Loss
sess = tf.Session()
sess.run(init)

# -------------------------------------------TRAINING & VALIDATION--------------------------------------------------
print('training...')

print('Fully connected Layer 1 weights', sess.run(conv1_weights))
print('Fully connected layer 1 weights shape', sess.run(conv1_weights).shape)

# Generate heat map for convolution layer 1 kernels
con1_wt = sess.run(conv1_weights)
print("Generating heatmaps")
heat_map(con1_wt)

training_loss = []  # Loss for plot
training_epoch = []  # Epoch count for plot
training_accuracy = []  # accuracy for plot

validation_loss = []  # Loss for plot
validation_epoch = []  # Epoch count for plot
validation_accuracy = []  # accuracy for plot

for step in range(1, num_steps + 1):

    print('Epoch number : ', step)

    batch_x, batch_y = batch_data(batch_start_index=batch_start, batch_end_index=batch_end, data_kind='train')

    # Forward and Back propogate the entire batch
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, cnn_mode: 'train'})

    # Training Loss for entire training data.
    batch_start_train = 0
    batch_end_train = batch_start_train + batch_size
    tr_loss, tr_acc = 0, 0  # Validation loss, validation accuracy for the entire validation data.
    tr_runs = 0  # Iterations to compute validation Loss

    while batch_end_train != batch_start_train:
        batch_x, batch_y = batch_data(batch_start_index=batch_start_train,
                                      batch_end_index=batch_end_train, data_kind='train')
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, cnn_mode: 'train'})
        tr_loss += loss
        tr_acc += acc

        # update batch start and end index
        batch_start_train = batch_end_train
        batch_end_train = min(batch_start_train + batch_end_train, train_data.shape[0])

        tr_runs += 1

    # Validation average accuracy across all batches.
    tr_acc /= tr_runs

    print("Step " + str(step) + ", Training Loss= " +
          "{:.4f}".format(tr_loss) + ", Training Accuracy= " + "{:.3f}".format(tr_acc))

    training_loss.append(tr_loss)
    training_accuracy.append(tr_acc)
    training_epoch.append(step)

    # Validation Loss for entire validation data in batches
    batch_start_valid = 0
    batch_end_valid = batch_start_valid + batch_size
    vl_loss, vl_acc = 0, 0  # Validation loss, validation accuracy for the entire validation data.
    vl_runs = 0  # Iterations to compute validation Loss

    while batch_end_valid != batch_start_valid:
        batch_x, batch_y = batch_data(batch_start_index=batch_start_valid,
                                      batch_end_index=batch_end_valid, data_kind='valid')
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, cnn_mode: 'valid'})
        vl_loss += loss
        vl_acc += acc

        # update batch start and end index
        batch_start_valid = batch_end_valid
        batch_end_valid = min(batch_start_valid + batch_end_valid, valid_data.shape[0])

        vl_runs += 1

    # Validation average accuracy across all batches.
    vl_acc /= vl_runs

    print("Step " + str(step) + ", Validation Loss= " +
          "{:.4f}".format(vl_loss) + ", Validation Accuracy= " + "{:.3f}".format(vl_acc))

    validation_loss.append(vl_loss)
    validation_accuracy.append(vl_acc)
    validation_epoch.append(step)

    # update batch start and end index
    batch_start = 0 if batch_end + batch_size > train_data.shape[0] else batch_end
    batch_end = batch_start + batch_size

    # update batch start and end index
    batch_start = 0 if batch_end + batch_size > train_data.shape[0] else batch_end
    batch_end = batch_start + batch_size

# Total Training loss and accuracy
train_total_accuracy = sum(training_accuracy) / len(training_accuracy)
train_total_loss = sum(training_loss)
print('Model trained with loss ' + str(train_total_loss) + ' and accuracy ' + str(train_total_accuracy))

# Total Validation loss and accuracy
valid_total_accuracy = sum(validation_accuracy) / len(validation_accuracy)
valid_total_loss = sum(validation_loss)
print('Model trained with loss ' + str(valid_total_loss) + ' and accuracy ' + str(valid_total_accuracy))

# Plot training & Validation loss
plt.figure(1)
plt.plot(training_epoch, training_loss, 'r-o', label="Training Loss")
plt.plot(training_epoch, validation_loss, 'g-o', label='Validation Loss')
plt.title('Training & Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../plots/loss.jpg')

# Plot training & Validation Accuracy
plt.figure(2)
plt.plot(training_epoch, training_accuracy, 'r-o', label='Training Accuracy')
plt.plot(training_epoch, validation_accuracy, 'g-o', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../plots/accuracy.jpg')

# Save CNN Model
saver = tf.train.Saver(tf.all_variables())
save = saver.save(sess, '../model/model.ckpt')
print('Model saved in file ', save)

# ----------------------------------------------------TESTING-------------------------------------------------------

test_predict = []  # list for storing class labels

batch_start = 0
batch_end = batch_start + batch_size
print('testing...')

while batch_start != batch_end:
    print('Test batch_end ', batch_end)
    batch_x = batch_data(batch_start_index=batch_start, batch_end_index=batch_end, data_kind='test')

    # Forward Prop and save predicted labels
    prediction = sess.run(test_pred_tensor, feed_dict={X: batch_x})
    test_predict.append(prediction.tolist())

    # update batch start and end index
    batch_start = batch_end
    batch_end = min(batch_start + batch_end, test_data.shape[0])

# Write test predictions to file
test_file = open('../data/test_predictions.txt', 'w')
test_file.write('id,label\n')

# Flatten 2d list to 1d
test_predict = sum(test_predict, [])

for index, label in enumerate(test_predict):
    text_to_write = str(index) + ',' + str(label) + '\n'
    test_file.write(text_to_write)

test_file.close()

sess.close()
