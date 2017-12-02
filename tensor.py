
#   TfLearn version of DeepMNIST
# taking from https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
# Create input object which reads data from MNIST datasets.  Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

savePath = 'model/LSTM_model.tfl'

# Using Interactive session makes it the default sessions so we do not need to pass sess 
sess = tf.InteractiveSession()

# Define placeholders for MNIST input data


    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")  
    #We now define the weights W and biases b for our model. 
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    #Before Variables can be used within a session, they must be initialized using that session
    sess.run(tf.global_variables_initializer())

    # Save the session for later use
    saver = tf.train.Saver()

# IMplement regression model
y = tf.nn.softmax(tf.matmul(x,W)+b)

# Set prediction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Set training
#TensorFlow has a variety of built-in optimization algorithms.
# For this example, we will use steepest gradient descent,
# with a step length of 0.5, to descend the cross entropy.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train the model
for _ in range(10000):
	batch = mnist.train.next_batch(100)  
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
	
# Evaluate the model 
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Print out the accuracy
acc_eval = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Current accuracy: %.2f%%"% (acc_eval*100))	

# Save the path to the trained model
saver.save(sess, savePath)
print('Session saved in path '+savePath)


#model.save('LSTM_model.tfl')
# model.save("mnist_model.h5")-





















# Reference to TFLearn library
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

image_rows = 28
image_cols = 28

# reshape the training and test images to 28 X 28 X 1 
train_images = mnist.train.images.reshape(mnist.train.images.shape[0],image_rows, image_cols, 1)
test_images =  mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

num_classes = 10
keep_prob = 0.5                 # fraction to keep (0-1.0)

# Define the shape of the data coming into the NN
input = input_data(shape=[None, 28, 28, 1], name='input')

# Do convolution on images, add bias and push through RELU activation
network = conv_2d(input, nb_filter=32, filter_size=3, activation='relu', regularizer="L2")
#   Notice name was not specified.  The name is defaulted to "Conv2D", and will be postfixed with "_n" 
#   where n is the number of the occurance.  Nice!
# take results and run through max_pool
network = max_pool_2d(network, 2)

saver = tf.train.Saver()

# 2nd Convolution layer
# Do convolution on images, add bias and push through RELU activation
network = conv_2d(network, nb_filter=64, filter_size=3, activation='relu', regularizer="L2")
# take results and run through max_pool
network = max_pool_2d(network, 2)

# Fully Connected Layer
network = fully_connected(network, 128, activation='tanh')

# dropout some neurons to reduce overfitting
network = dropout(network, keep_prob)

# Readout layer
network = fully_connected(network, 10, activation='softmax')

# Set loss and measurement, optimizer
network = regression(network, optimizer='adam', learning_rate=0.01,
                        loss='categorical_crossentropy', name='target')

# Training
num_epoch = 1   # number of times through the data.
model = tflearn.DNN(network, tensorboard_verbose=0)  # for more info in tenorboard turn on tensorboard_verbose
model.fit({'input': train_images}, {'target': mnist.train.labels}, n_epoch=num_epoch,
           validation_set=({'input': test_images}, {'target': mnist.test.labels}),
            show_metric=True, run_id='TFLearn_DeepMNIST')
		
#model.save('LSTM_model.tfl')
# model.save("mnist_model.h5")-


# Save the path to the trained model
saver.save(sess, savePath)
print('Session saved in path '+savePath)