import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

a = tf.constant(2.0)
b = tf.constant(3.0)

print(a)
print(b)

sess = tf.InteractiveSession()
print(sess.run([a,b]))

s = a * b
print(s)
print(sess.run(s))

c = tf.constant(8.0)
s = tf.constant(6.0)
d = tf.constant(4.0)

a_plus_b = a+b
m = a_plus_b * c
n = d + s
sub = -s * n
ans = m + sub

print(sess.run(ans))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a + b

print(sess.run(adder, feed_dict = {a: 4.0, b: 10.0}))

# HOFs in Python

input_list = list(np.arange(1,10))
print(input_list)

square = a * a

squares = map(lambda x: x*x, input_list)
print(squares)

output_list = [sess.run(square, feed_dict = {a: i}) for i in input_list]

print(output_list)

plt.plot(input_list, output_list)
plt.title("Squares")
plt.xlabel("inputs")
plt.ylabel("outputs")
plt.show()

W = tf.Variable(tf.constant(0.1, shape=[10,1]))

x = tf.placeholder(tf.float32, shape = [1,10])

mat_mul = tf.matmul(x, W)

data = np.arange(10).reshape((1,10))

init = tf.global_variables_initializer()
print(sess.run(init))
print(sess.run(mat_mul, feed_dict = {x: data}))
print(sess.run(W))






