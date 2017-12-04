
#   TfLearn version of DeepMNIST
# taking from https://www.tensorflow.org/get_started/mnist/pros
from tensorflow.examples.tutorials.mnist import input_data
# Create input object which reads data from MNIST datasets.  Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

savePath = 'tmp/tensor_model'

# Using Interactive session makes it the default sessions so we do not need to pass sess 
import tensorflow as tf
sess = tf.InteractiveSession()



# Define placeholders for MNIST input data

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])  
    #We now define the weights W and biases b for our model. 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Before Variables can be used within a session, they must be initialized using that session
sess.run(tf.global_variables_initializer())

# Save the session for later use
saver = tf.train.Saver()

# regression model
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

#print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
   # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))
#model.save('LSTM_model.tfl')
# model.save("mnist_model.h5")-