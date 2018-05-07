import sys
sys.path.append('/run/media/toloan/0.0/programming/AI/Email/logistic/databuilder')
import datacreate as dtc
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_path = "..//..//lingspam_public//bare//part1"
dict_path = "..//..//lingspam_public//bare//part1"
test_path = "..//..//lingspam_public//bare//part2"
train_X,train_Y = dtc.training_set(dict_path,train_path)
train_X=numpy.asarray(train_X)
train_Y=numpy.asarray(train_Y)

n_sample=train_X.shape[0]
n=train_Y.shape[1]

X = tf.placeholder("int",shape=(1,n))
Y = tf.placeholder("float")

b = tf.Variable(rng.randn(), name = "bias")
W = tf.Variable([rng.randn() for i in range(n)], name = "weight")

h = tf.divide(1,tf.add(1,tf.pow(tf.exp(tf.add(b,tf.matmul(W,X,transpose_b=True))))))
likelihood = tf.reduce_sum(Y*tf.log(h)+(1-Y)*tf.log(1-h))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).maximize(likelihood)


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		for (x,y) in zip(train_X,train_Y):
			sess.run(optimizer,feed_dict={X: x, Y: y})

		if (epoch+1)%display_step == 0:
			c = sess.run(likelihood,feed_dict={X:train_X,Y: train_Y})
			print("epoch:",'%04d' % (epoch+1), "cost=",'{:.9}'.format(c))

	print("Optimization Finished")
	training_likelihood = sess.run(likelihood,feed_dict={X:train_X,Y:train_Y})
	print("training_likelihood:",training_likelihood)

	plt.plot(train_X,train_Y, 'ro', label='Original data')
	# plt.plot(train_X,sess.run(h), label='Fitted line')
	plt.legend()
	plt.show()

	test_X,test_Y = dtc.training_set(dict_path,test_path)
	test_X=numpy.asarray(test_X)
	test_Y=numpy.asarray(test_Y)

	testing_likelihood = sess.run(likelihood,feed_dict = {X:test_X, Y:test_Y})
	print("testing_likelihood ",testing_likelihood)
	print("|testing_likelihood-cost_likelihood|",abs(training_likelihood-testing_likelihood))

	plt.plot(test_X, test_Y, 'bo', label="Testing data")
	plt.plot(train_X, sess.run(h), label='Fitted line')
	plt.legend()
	plt.show()
	print("done")	

			


