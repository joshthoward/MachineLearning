import tensorflow as tf, time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weightVariable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def biasVariable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Input and Output place holders
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
IM = tf.reshape(x, [-1,28,28,1])

# Convolutional Layers
cO1 = 32
cO2 = 64

cW1 = weightVariable([5, 5,  1,  cO1])
cb1 = biasVariable([cO1])

cW2 = weightVariable([5, 5, cO1, cO2])
cb2 = biasVariable([cO2])

H1 = maxPool(tf.nn.relu(conv(IM, cW1) + cb1))
H2 = maxPool(tf.nn.relu(conv(H1, cW2) + cb2))

# Fully Connected Layers
fW1 = weightVariable([7*7*64, 1024])
fb1 = biasVariable([1024])

fW2 = weightVariable([1024, 10])
fb2 = biasVariable([10])

H3 = tf.nn.relu(tf.matmul(tf.reshape(H2, [-1, 7*7*64]), fW1) + fb1)

d_prob = tf.placeholder(tf.float32)
y_conv = tf.nn.softmax(tf.matmul(tf.nn.dropout(H3, d_prob), fW2) + fb2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as S:

    S.run(tf.initialize_all_variables())
    start = time.time()
    
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], d_prob: 1.0})
        print("step %d, training accuracy %g, elapsed time %g"%(i, train_accuracy, time.time()-start))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], d_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, d_prob: 1.0}))



