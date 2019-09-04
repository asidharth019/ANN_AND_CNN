import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_file_path = '../data/train.csv'
valid_file_path = '../data/val.csv'
test_file_path = '../data/test.csv'

# Training Parameters
learning_rate = 0.001
num_steps = 150
batch_size = 4000
batch_start = 0
batch_end = batch_start + batch_size
display_step = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.6  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


# load training, valid, test data
def load_data(file_path, data_kind):
    # data_kind specify train/valid/test data
    print('loading ' + data_kind + ' data...')
    file = np.genfromtxt(file_path, dtype=float, delimiter=',')

    file = np.delete(file, 0, axis=0)  # delete first row
    file = np.delete(file, 0, axis=1)  # delete first column

    if data_kind == 'train' or data_kind == 'val':
        data = file[:, 0:-1]
        label = file[:, -1]
        return data, label
    else:  # test data
        data = file
        return data


# MNIST Fashion data
tf.set_random_seed(1)
train_data, train_label = load_data(file_path=train_file_path, data_kind='train')
valid_data, valid_label = load_data(file_path=valid_file_path, data_kind='val')
test_data = load_data(file_path=test_file_path, data_kind='test')


def cnn_model_fn(features):
    """Model function for CNN."""
    print("features shape", features.shape)

    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    print(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="same")
    print(pool1)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    print(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="same")
    print(pool2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    print(conv3)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    print(conv4)
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding="same")
    print(pool3)
    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 256])
    print(pool3_flat)
    fc1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    print(fc1)
    fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
    print(fc2)
    fc2_bn = tf.nn.batch_normalization(x=fc2, mean=0, variance=1, scale=1, offset=0, variance_epsilon=1e-6)
    print(fc2_bn)
    fc3 = tf.layers.dense(inputs=fc2_bn, units=10)
    print(fc3)
    return fc3


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

# Compute Training, Validation, Testing Loss
with tf.Session() as sess:
    sess.run(init)

    # --------------------------------------------------TRAINING--------------------------------------------------------
    print('training...')

    training_loss = []  # Loss for plot
    training_epoch = []  # Epoch count for plot
    training_accuracy = []  # accuracy for plot

    for step in range(1, num_steps+1):
        batch_x, batch_y = batch_data(batch_start_index=batch_start, batch_end_index=batch_end, data_kind='train')

        # Forward and Back propogate the entire batch
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:  # print training loss after every display_step iterations
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

            training_loss.append(loss)
            training_accuracy.append(acc)
            training_epoch.append(step)

        # update batch start and end index
        batch_start = 0 if batch_end + batch_size > train_data.shape[0] else batch_end
        batch_end = batch_start + batch_size

    # Total Training loss and accuracy
    train_total_accuracy = sum(training_accuracy) / len(training_accuracy)
    train_total_loss = sum(training_loss)
    print('Model trained with loss ' + str(train_total_loss) + ' and accuracy ' + str(train_total_accuracy))

    # Plot training loss
    plot_figure(x_axis=training_epoch, y_axis=training_loss, x_label='Epoch', y_label='Training Loss',
                title='Training Loss Vs Epoch', figure_num=1)

    # Plot training accuracy
    plot_figure(x_axis=training_epoch, y_axis=training_accuracy, x_label='Epoch', y_label='Training Accuracy',
                title='Training Accuracy Vs Epoch', figure_num=2)

    # ---------------------------------------------------VALIDATION-----------------------------------------------------
    print('validating...')

    validation_loss = []  # Loss for plot
    validation_epoch = []  # Epoch count for plot
    validation_accuracy = []  # accuracy for plot

    batch_start = 0
    batch_end = batch_start + batch_size

    for step in range(1, num_steps+1):
        batch_x, batch_y = batch_data(batch_start_index=batch_start, batch_end_index=batch_end, data_kind='valid')

        if step % display_step == 0 or step == 1:  # print training loss after every display_step iterations
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))

            validation_loss.append(loss)
            validation_accuracy.append(acc)
            validation_epoch.append(step)

        # update batch start and end index
        batch_start = 0 if batch_end + batch_size > valid_data.shape[0] else batch_end
        batch_end = batch_start + batch_size

    # Total Validation loss and accuracy
    valid_total_accuracy = sum(validation_accuracy) / len(validation_accuracy)
    valid_total_loss = sum(validation_loss)
    print('Model trained with loss ' + str(valid_total_loss) + ' and accuracy ' + str(valid_total_accuracy))

    # Plot Validation loss
    plot_figure(x_axis=validation_epoch, y_axis=validation_loss, x_label='Epoch', y_label='Validation Loss',
                title='Validation Loss Vs Epoch', figure_num=1)

    # Plot validation accuracy
    plot_figure(x_axis=validation_epoch, y_axis=validation_accuracy, x_label='Epoch', y_label='Validation Accuracy',
                title='Validation Accuracy Vs Epoch', figure_num=2)

    # ----------------------------------------------------TESTING-------------------------------------------------------

    test_predict = []  # list for storing class labels

    batch_start = 0
    batch_end = batch_start + batch_size
    print('testing...')

    while batch_start != batch_end:
        print('Test batch_end ', batch_end)
        batch_x = batch_data(batch_start_index=batch_start, batch_end_index=batch_end, data_kind='test')

        # Forward Prop and save predicted labels
        test_predict.append(sess.run(test_pred_tensor, feed_dict={X: batch_x}))

        # update batch start and end index
        batch_start = batch_end
        batch_end = min(batch_start + batch_end, test_data.shape[0])

    # Write test predictions to file
    test_file = open('../data/test_predictions.txt', 'w')
    test_file.write('id,label\n')

    # Flatten 2d list to 1d
    test_predict = np.array(test_predict)
    test_predict = test_predict.flatten()
    test_predict = test_predict.tolist()

    for index, label in enumerate(test_predict):
        text_to_write = str(index) + ',' + str(label) + '\n'
        test_file.write(text_to_write)

    test_file.close()




