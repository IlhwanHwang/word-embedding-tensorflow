import tensorflow as tf
import numpy as np
from argparse import Namespace

meta = Namespace()
meta.batch_size = 128
meta.embed_dim = 128
meta.context_size = 10
meta.noise_size = 64
meta.learning_rate = 1e-2
meta.training_step = 10000

def build_data(filename):
	f = open(filename, 'r')
	content = f.readline().lower().split()
	domain = list(set(content))
	domain_dic = {key : index for index, key in enumerate(domain)}

	samples = []
	for i in range(len(content)):
		for j in range(-meta.context_size // 2, meta.context_size // 2):
			if i + j < 0 or i + j >= len(content) or j == 0:
				continue
			samples.append([domain_dic[content[i]], domain_dic[content[i + j]]])

	return domain, np.array(samples, dtype=np.int32)

domain, samples = build_data('data.txt')
domain_size = len(domain)

print('{} samples available.'.format(len(samples)))

def xavier_variable(name, shape):
	if len(shape) == 1:
		inout = np.sum(shape) + 1
	else:
		inout = np.sum(shape)

	init_range = np.sqrt(6.0 / inout)
	initializer = tf.random_uniform_initializer(-init_range, init_range)
	return tf.get_variable(name, shape, initializer=initializer)

W = xavier_variable('W', shape=[domain_size, meta.embed_dim])
B = xavier_variable('B', shape=[domain_size])

inputs = tf.placeholder(tf.int32, shape=[meta.batch_size])
outputs = tf.placeholder(tf.int32, shape=[meta.batch_size, 1])
noises = tf.random_uniform([meta.batch_size, meta.noise_size], minval=0, maxval=domain_size, dtype=tf.int32)

input_embed = tf.nn.embedding_lookup(W, inputs)
noise_embed = tf.nn.embedding_lookup(W, noises)

loss = tf.reduce_mean(tf.nn.nce_loss(W, B, outputs, input_embed, meta.noise_size, domain_size))
train = tf.train.GradientDescentOptimizer(meta.learning_rate).minimize(loss)
# train = tf.train.AdamOptimizer(meta.learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(meta.training_step):
	indices = np.random.choice(domain_size, size=[meta.batch_size])
	batch = samples[indices]
	feed = {
		inputs : batch[:,0],
		outputs : np.reshape(batch[:,1], (-1, 1))
	}
	_, loss_value = sess.run([train, loss], feed_dict=feed)

	print(loss_value)

W_embed = sess.run(W, feed_dict={})

np.save('embed.npy', W_embed)

A = 'he'
B = 'his'
C = 'she'

a = W_embed[domain.index(A)]
b = W_embed[domain.index(B)]
c = W_embed[domain.index(C)]

d = np.argmax(np.matmul(b - a + c, W_embed.T))

print('{} to {} is {} to {}.'.format(A, B, C, domain[d]))

