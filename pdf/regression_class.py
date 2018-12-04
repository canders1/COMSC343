#import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
This tutorial from the AI at UCLA's technical blog post:
https://uclaacmai.github.io/Linear-Regression
"""

testlines = []
testans = []
trainlines = []
trainans = []

for line in open("boston2.txt", 'r').readlines()[0:100]:
	tl = line.strip('\n').strip(' ').split('  ')
	testlines.append(map(lambda x:float(x.strip(' ')),tl[0:13]))
	testans.append([float(tl[13].strip(' '))])

for line in open("boston2.txt", 'r').readlines()[100:]:
	tl = line.strip('\n').strip(' ').split('  ')
	trainlines.append(map(lambda x:float(x.strip(' ')),tl[0:13]))
	trainans.append([float(tl[13].strip(' '))])

X_train = np.array(trainlines, dtype=np.float32)
X_test = np.array(testlines, dtype=np.float32)
Y_train = np.array(trainans, dtype=np.float32)
Y_test = np.array(testans, dtype=np.float32)

print(",".join([str(t.shape) for t in (X_train, X_test, Y_train, Y_test)]))

prices = Y_train.tolist()
student_teacher_ratios = [X_train[i][10] for i in range(X_train.shape[0])]
plt.scatter(student_teacher_ratios,prices)
plt.show()

X = tf.placeholder(tf.float32,shape=[None,13])
Y = tf.placeholder(tf.float32, shape = [None,1])

W = tf.Variable(tf.constant(0.1,shape=[13,1]))
b = tf.Variable(tf.constant(0.1))

y_pred = tf.matmul(X,W) + b
loss = tf.reduce_mean(tf.square(y_pred - Y))
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
initial_loss = loss.eval(feed_dict = {X:X_train, Y:Y_train})

print("initial loss: {}".format(initial_loss))

for i in range(100):
	#Run the optimization step with training data
	sess.run(opt, feed_dict = {X:X_train, Y:Y_train})
	print("epoch "+str(i)+"loss:{}".format(loss.eval(feed_dict = {X:X_train, Y:Y_train})))

final_test_loss = loss.eval(feed_dict = {X:X_test,Y:Y_test})
print("final loss (test): {}".format(final_test_loss))



