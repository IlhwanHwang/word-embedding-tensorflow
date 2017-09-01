#
# Word Embedding Solver using TensorFlow
# Author Ilhwan Hwang (https://github.com/IlhwanHwang)
# Date 17.09.02
#

import tensorflow as tf
import numpy as np
from argparse import Namespace

meta = Namespace()
meta.batch_size = 128
meta.embed_dim = 512
meta.window_size = 2
meta.noise_size = 64
meta.learning_rate = 1e0
meta.training_step = 1000000
meta.interval_save = 100000
meta.interval_print = 100
meta.interval_test = 10000
meta.domain_size = 100000
meta.savefile = './model/model.ckpt'
meta.data_dir = './data'
meta.file_data = '../text8'
meta.test_data = '../questions-words.txt'
meta.label_unknown = '*UNKNOWN*'

class Word2Vec:

	def __init__(self, meta):
		try:
			print('Loding domain information...')
			self.content = np.load(meta.data_dir + '/content.npy')
			self.content_contrib = np.load(meta.data_dir + '/content_contrib.npy')
			self.domain = np.load(meta.data_dir + '/domain.npy')
			self.noise_domain = np.load(meta.data_dir + '/noise_domain.npy')
		except IOError:
			self.build_data(meta)

		self.invdomain = {word : ind for ind, word in enumerate(self.domain)}
		self.unknown_ind = self.invdomain[meta.label_unknown]


	def build_data(self, meta):
		f = open(meta.file_data, 'r', encoding='utf-8')
	
		print('Reading content...')
		
		content_raw = f.readline().split()
		content_raw = np.array(content_raw)

		print('Content length is {}.'.format(len(content_raw)))
		print('Counting domain...')

		domain, content_ind, count = np.unique(content_raw, return_inverse=True, return_counts=True)

		print('Raw domain size is {}.'.format(domain.size))

		# Restrict domain size. Words with too small counts will be merged into word *UNKNOWN*
		if domain.size > meta.domain_size:
			print('Cropping domain...')

			ind_least_words = np.argpartition(count, domain.size - meta.domain_size + 1)[:-meta.domain_size + 1]
			content_raw[np.in1d(content_ind, ind_least_words)] = meta.label_unknown
			
			print('Re-counting domain...')

			domain, content_ind, count = np.unique(content_raw, return_inverse=True, return_counts=True)

		meta.domain_size = domain.size
		print('Domain size is {}.'.format(len(domain)))

		# Subsampling is slow. Instead, diminish their contribution to the total loss.
		freq = count / len(domain)
		subsample_ratio = 1. - np.sqrt(1e-5 / freq)
		content_contrib = subsample_ratio[content_ind]

		# Sampling noises, we will draw samples with probability raised to the power of 2/3.
		noise_count = np.power(count, 2./3.)
		# Wait, weighted random is also slow. Just draw from large pool that approximates the probability distribution.
		noise_domain = np.repeat(np.arange(domain.size), noise_count.astype(np.int32))

		self.content = content_ind
		self.content_contrib = content_contrib
		self.domain = domain
		self.noise_domain = noise_domain

		# Save them for later.
		np.save(meta.data_dir + '/content.npy', self.content)
		np.save(meta.data_dir + '/content_contrib.npy', self.content_contrib)
		np.save(meta.data_dir + '/domain.npy', self.domain)
		np.save(meta.data_dir + '/noise_domain.npy', self.noise_domain)


	def word_index(self, word):
		return np.where(self.domain == word)[0]


	def index_word(self, index):
		return self.domain[index]


	def nearest(self, Q, word, k):
		ind = self.word_index(word)[0]
		embed = Q[ind]
		dot = np.matmul(Q, np.array([embed]).T)
		return self.domain[np.argpartition(-np.reshape(dot, [-1]), k)[0:k]]


	def analogy(self, Q, a, b, c, k):
		a_ind = self.word_index(a)[0]
		b_ind = self.word_index(b)[0]
		c_ind = self.word_index(c)[0]

		a_embed = np.array([Q[a_ind]])
		b_embed = np.array([Q[b_ind]])
		c_embed = np.array([Q[c_ind]])

		return self.nearest_embed(Q, b_embed - a_embed + c_embed, k)


	def test(self, Q, filename):
		score = 0
		total = 0
		with open(filename, 'r') as f:
			for line in f:
				try:
					a, b, c, d = line.split()
				except ValueError:
					continue

				d_pred = self.analogy(Q, a, b, c, 1)
				if d == d_pred[0]:
					score += 1

				total += 1

		return score / total

	def build_graph(self, meta):

		# Building graph for training.
		def xavier_variable(name, shape):
			if len(shape) == 1:
				inout = np.sum(shape) + 1
			else:
				inout = np.sum(shape)

			init_range = np.sqrt(6.0 / inout)
			initializer = tf.random_uniform_initializer(-init_range, init_range)
			return tf.get_variable(name, shape, initializer=initializer)

		# Q is embedding vector.
		Q = xavier_variable('Q', shape=[meta.domain_size, meta.embed_dim])
		# R is weight vector.
		R = xavier_variable('R', shape=[meta.domain_size, meta.embed_dim])
		# B is bias.
		B = xavier_variable('B', shape=[meta.domain_size])

		# Qhat is unit embedding vector.
		Qhat = tf.nn.l2_normalize(Q, dim=1)

		# There is no placeholder. Drawing samples from CPU is slow. GPU 'em all!
		content = tf.constant(self.content, dtype=tf.int32)
		content_contrib = tf.constant(self.content_contrib, dtype=tf.float32)
		noise_domain = tf.constant(self.noise_domain, dtype=tf.int32)

		# w_data is 'word from data'. ind_w_data is word's index in content.
		# Draw words from content randomly, and repeat them. They will used with all context surrounding.
		ind_w_data = tf.tile(
			tf.random_uniform(
				shape=[meta.batch_size], 
				minval=meta.window_size, 
				maxval=self.content.size - meta.window_size, 
				dtype=tf.int32), 
			[meta.window_size * 2])
		
		# c_data is 'context from data'.
		# Start from numpy array, as tensorflow doesn't provide 'repeat' layer.
		window_range = np.concatenate([np.arange(1, meta.window_size + 1), -np.arange(1, meta.window_size + 1)])
		ind_c_data = ind_w_data + tf.constant(np.repeat(window_range, meta.batch_size))

		# Look up the table with given index.
		w_data = tf.nn.embedding_lookup(content, ind_w_data)
		c_data = tf.nn.embedding_lookup(content, ind_c_data)
		loss_contrib_data = tf.nn.embedding_lookup(content_contrib, ind_w_data)

		# Noises are just random and drawn from noise domain.
		ind_w_noise = tf.random_uniform(
			shape=[meta.batch_size * meta.window_size * 2 * meta.noise_size], 
			minval=0, 
			maxval=self.noise_domain.size, 
			dtype=tf.int32)
		ind_c_noise = tf.random_uniform(
			shape=[meta.batch_size * meta.window_size * 2 * meta.noise_size], 
			minval=0, 
			maxval=self.noise_domain.size, 
			dtype=tf.int32)

		w_noise = tf.nn.embedding_lookup(self.noise_domain, ind_w_noise)
		c_noise = tf.nn.embedding_lookup(self.noise_domain, ind_c_noise)

		# Extract embedding vectors and weights and biases given words and contexts.
		q_w_data = tf.nn.embedding_lookup(Q, w_data)
		r_c_data = tf.nn.embedding_lookup(R, c_data)
		b_c_data = tf.nn.embedding_lookup(B, c_data)

		q_w_noise = tf.nn.embedding_lookup(Q, w_noise)
		r_c_noise = tf.nn.embedding_lookup(R, c_noise)
		b_c_noise = tf.nn.embedding_lookup(B, c_noise)

		score_data = tf.reduce_sum(tf.multiply(q_w_data, r_c_data), axis=1) + b_c_data
		score_noise = tf.reduce_sum(tf.multiply(q_w_noise, r_c_noise), axis=1) + b_c_noise

		# log-sigmoid is exactly same as sigmoid_cross_entropy_with_logits + ones or zeros
		self.loss = -(
			tf.reduce_sum(
				tf.multiply(
					tf.log(
						tf.sigmoid(score_data)
						), 
					loss_contrib_data) # Given data's contribution to the total loss.
				) + 
			tf.reduce_sum(
				tf.log(
					tf.sigmoid(-score_noise)
					)
				)
			) / meta.batch_size / meta.window_size / 2

		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.train = tf.train.GradientDescentOptimizer(meta.learning_rate).minimize(self.loss, self.global_step)

		# Building graph for testing.
		# Analogy test.
		self.a_data = tf.placeholder(tf.int32, [None, ])
		self.b_data = tf.placeholder(tf.int32, [None, ])
		self.c_data = tf.placeholder(tf.int32, [None, ])

		a_embed = tf.nn.embedding_lookup(Qhat, self.a_data)
		b_embed = tf.nn.embedding_lookup(Qhat, self.b_data)
		c_embed = tf.nn.embedding_lookup(Qhat, self.c_data)

		self.d_pred = tf.argmax(
			tf.matmul(b_embed - a_embed + c_embed, tf.transpose(Qhat)), 
			axis=1)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state('./model')
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			self.sess.run(tf.global_variables_initializer())


	def train_step(self):
		_, loss_value = self.sess.run([self.train, self.loss], feed_dict={})
		step = self.sess.run(self.global_step)
		return step, loss_value


	def test_analogy(self, data_word):
		data = np.array(
			[[self.invdomain[a], self.invdomain[b], self.invdomain[c], self.invdomain[d]] for a, b, c, d in data_word])

		feed = {
			self.a_data : data[:,0],
			self.b_data : data[:,1],
			self.c_data : data[:,2],
		}
		answer = self.sess.run(self.d_pred, feed_dict=feed)
		correct = (answer == data[:,3])

		# Print the first one as sample.
		rate = np.count_nonzero(correct) / correct.size
		print(
			'Batch accuracy: {:.2f}%, {}->{} :: {}->{}'.format(
				rate * 100, 
				self.domain[data[0,0]], 
				self.domain[data[0,1]], 
				self.domain[data[0,2]], 
				self.domain[answer[0]]
				)
			)

		return rate


	def test_batch(self, filename):
		batch_size = 128
		batch = []
		total = 0
		total_div = 0

		with open(filename, 'r') as f:
			print('Testing {}...'.format(filename))
			count = 0
			
			for line in f:
				data = line.lower().split()
				if data[0] == ':':
					continue

				batch.append(data)

				if len(batch) >= batch_size:
					try:
						rate = self.test_analogy(batch)
						total += rate
						total_div += 1
					except KeyError:
						print('Illegal words, passing batch.')
						pass
					batch = []
					
		print('Total accuracy is {:.2f}%.'.format(total / total_div * 100))


	def save(self, filename):
		self.saver.save(self.sess, filename, global_step=self.global_step)


if __name__ == '__main__':

	w2v = Word2Vec(meta)
	w2v.build_graph(meta)

	for _ in range(meta.training_step):
		step, loss = w2v.train_step()

		if step % meta.interval_print == 0:
			print('{}: Loss {:g}'.format(step, loss))

		if step % meta.interval_test == 0:
			try:
				w2v.test_batch(meta.test_data)
			except IOError:
				pass

		if step % meta.interval_save == 0:
			w2v.save(meta.savefile)
			print('Saving status...')
			
