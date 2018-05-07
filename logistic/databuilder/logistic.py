import sys
import datacreate as dtc
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

learning_rate = 0.0001
training_epochs = 1000
display_step = 50

train_path = "..//..//lingspam_public//bare//part1"
dict_path = "..//..//lingspam_public//bare//part1"
test_path = "..//..//lingspam_public//bare//part2"
train_X,train_Y = dtc.training_set(dict_path,train_path)
train_X=numpy.asarray(train_X)
train_Y=numpy.asarray(train_Y)

n_sample=train_X.shape[0]
n=train_X.shape[1]
threadhold=tf.constant(0.5)

X = tf.placeholder("float",shape=(None,n))
Y = tf.placeholder("float")

b = tf.Variable(rng.randn(), name = "bias")
W = tf.Variable(tf.zeros([1, n]), name = "weight")

z=tf.add(b,tf.matmul(tf.cast(W, tf.float32),X,transpose_b=True))
h=tf.sigmoid(z)
#likelihood= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=X, labels=Y))
likelihood = tf.reduce_sum(Y*tf.log(h)+(1-Y)*tf.log(1-h))
neg_likelihood=-1*likelihood

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(neg_likelihood)


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		for (x,y) in zip(train_X,train_Y):
			sess.run(optimizer,feed_dict={X: numpy.asmatrix(x), Y: numpy.asmatrix(y)})

	print("Optimization Finished")
	training_likelihood = sess.run(likelihood,feed_dict={X:train_X,Y:train_Y})
	print("training_likelihood:",training_likelihood)

	prediction = tf.abs(h-Y)
	correct_prediction = tf.cast(tf.less(prediction,threadhold),tf.int32)
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
	test_X,test_Y = dtc.training_set(dict_path,test_path)
	test_X=numpy.asarray(test_X)
	test_Y=numpy.asarray(test_Y)
	for (x,y) in zip(test_X,test_Y):
		print("h: ",sess.run(h,feed_dict={X: test_X, Y: test_Y}))
		print("predict: ",sess.run(prediction,feed_dict={X: test_X, Y: test_Y}))
		print("correct predict: ",sess.run(correct_prediction,feed_dict={X: test_X, Y: test_Y}))

	print("Accuracy:", sess.run(accuracy,feed_dict={X:numpy.asmatrix(test_X),Y:numpy.asmatrix(test_Y)}))

	print("done")	

			


