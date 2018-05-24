import datacreate as dtc 
import tensorflow as tf 
import matplotlib.pyplot  as plt 
import numpy
import time
from cvxopt import matrix,solvers

dict_path = "..//..//lingspam_public//bare//part1"
train_path = "..//..//lingspam_public//bare//part1"
test_path = "..//..//lingspam_public//bare//part6"

spam_x, spam_y = dtc.spam_set_nltk(dict_path,train_path)
spam_x = numpy.array(spam_x)
spam_y = numpy.array(spam_y)
print("spam training data: ",spam_x.shape[0])

not_spam_x, not_spam_y = dtc.not_spam_set_nltk(dict_path,train_path)
not_spam_x = numpy.asarray(not_spam_x)
not_spam_y = numpy.asarray(not_spam_y)

train_x = numpy.concatenate((spam_x,not_spam_x))
train_y = numpy.concatenate((spam_y,not_spam_y))
print("not spam training data: ",spam_x.shape[0])

N=train_x.shape[1]
sample =train_x.shape[0]

start_time = time.time()
V=numpy.concatenate((spam_x,-not_spam_x)).T
K = matrix(V.T@V,(sample,sample),'d')
p = matrix(-numpy.ones((sample,1)))
G = matrix(-numpy.eye(sample))
h = matrix(numpy.zeros((sample,1)))

A = matrix(train_y.T,(1,sample),'d')
#print(A.size)

b = matrix(numpy.zeros((1,1)),(1,1),'d')

solvers.options['show_progress'] = False
sol = solvers.qp(K,p,G,h,A,b)

l = numpy.asarray(sol['x'])
#print('lambda = ',l.T)

epsilon= 1e-6
S = numpy.where(l > epsilon)[0]
VS = numpy.asarray([V[:,s] for s in S])
XS = numpy.asarray([train_x[s,:] for s in S ])
yS = numpy.asarray([train_y[s] for s in S])
lS = matrix(numpy.asarray([l[s] for s in S]))

w = VS.T.dot(lS)# 24342
b = numpy.mean(yS.T-w.T@XS.T)
print("learning time: ",time.time()-start_time)
print("ads")

check = 0
test_x,test_y = dtc.training_set_nltk(dict_path,test_path)
test_x=numpy.asarray(test_x)
test_y=numpy.asarray(test_y)
n_test =test_x.shape[0]
learning_time = time.time()
predict=[]
precise_ham_sum=0
precise_spam_sum=0
precise_ham=0
precise_spam=0
recall_ham = 0
recall_spam = 0
for (x,y) in zip(test_x,test_y):
	#print(y)
	#print()
	if y==1:
		recall_spam =recall_spam+ 1
	else:
		recall_ham = recall_ham + 1
	p=w.T@x+b
	if p >= 0: 
		precise_spam_sum = precise_spam_sum+1
	else:
		precise_ham_sum = precise_ham_sum + 1
	if  p*y>= 0:
		check = check + 1
		if p>=0:
			precise_spam = precise_spam +1
		else: 
			precise_ham = precise_ham+1
print("return time : ",time.time()-learning_time)
print(check)
print(n_test)
print("accuracy: ",check/n_test)
print("precision: ", precise_spam/precise_spam_sum, " ", precise_ham/precise_ham_sum)
print("recall:", precise_spam/recall_spam, " ", precise_ham/recall_ham)




