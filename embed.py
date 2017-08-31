import tensorflow as tf
import numpy as np
from argparse import Namespace

meta = Namespace()
meta.batch_size = 128
meta.embed_dim = 128
meta.context_size = 10
meta.noise_size = 64
meta.learning_rate = 1e-2
meta.training_step = 100000
meta.interval_save = 20000
meta.interval_print = 100
meta.domain_size = 50000
meta.savefile = './model/model.ckpt'
meta.file_domain = './data/domain.npy'
meta.file_samples = './data/samples.npy'
meta.file_noise_domain = './data/noise_domain.npy'
meta.file_data = '../text8'

class Dataset:

	def __init__(self, meta, filename):
		f = open(filename, 'r')
	
		print('Reading content...')
		
		content_raw = f.readline().lower().split()

		print('Counting domain...')
		domain, content_ind, count = np.unique(content_raw, return_inverse=True, return_counts=True)

		print('Domain size is {}.'.format(len(domain)))
		print('Content length is {}.'.format(len(content)))

		freq = count / len(domain)
		subsample_ratio = 1. - np.sqrt(1e-5 / freq)
		content_pd = subsample_ratio[content_ind[:,0]]
		content_pd = content_pd / np.sum(content_pd)

		noise_count = np.power(count, 2./3.)
		noise_pd = noise_count / np.sum(noise_count)

		self.content = content_ind
		self.content_pd = content_pd
		self.domain = domain
		self.noise_pd = noise_pd
		self.subsample_map = subsample_map


	def batch(self, batch_size, noise_size, window_size):

		ind_w_data = np.random.choice(np.arange(window_size, self.domain.size - window_size), size=[batch_size], p=self.content_pd[window_size:-window_size])
		w_data = self.content[ind_w_data]
		ind_c_data = ind_w_data + np.multiply(np.random.randint(1, window_size + 1, size=[batch_size]), np.random.choice([1, -1], size=[batch_size], replace=False))
		c_data = self.content[ind_c_data]

		w_noise = np.random.choice(np.arange(self.domain.size), size=[batch_size * noise_size], p=self.noise_pd)
		c_noise = np.random.choice(np.arange(self.domain.size), size=[batch_size * noise_size], p=self.noise_pd)

		return w_data, c_data, w_noise, c_noise


def build_data(filename):
	f = open(filename, 'r')
	
	print('Reading content...')
	
	content = f.readline().lower().split()

	print('Counting domain...')
	domain, content_ind, count = np.unique(content, return_inverse=True, return_counts=True)

	print('Domain size is {}.'.format(len(domain)))
	print('Content length is {}.'.format(len(content)))

	freq = count / len(domain)
	subsample_ratio = 1. - np.sqrt(1e-5 / freq)

	print('Sampling...')

	samples = np.array([], dtype=np.int32)
	for ofs in range(-meta.context_size // 2, meta.context_size // 2):
		if ofs > 0:
			sample = np.transpose(np.concatenate([[content_ind[0:-ofs]], [content_ind[ofs:]]], axis=0))
		if ofs < 0:
			sample = np.transpose(np.concatenate([[content_ind[-ofs:]], [content_ind[0:ofs]]], axis=0))
		if ofs == 0:
			continue

		if samples.size:
			samples = np.concatenate([samples, sample])
		else:
			samples = np.array(sample, dtype=np.int32)

		print('Processing {} samples...'.format(samples.shape[0]))

	print('Subsampling...')
	noise = np.random.random([samples.shape[0]])
	threshold = subsample_ratio[samples[:,0]]
	samples = samples[noise < threshold]

	print('Generating noise domain...')
	noise_count = np.power(count, 2./3.).astype(np.int64)
	noise_domain = np.repeat(np.arange(len(domain)), noise_count)

	print('Sample size is {}.'.format(len(samples)))

	return domain, samples, noise_domain

try:
	print('Loding domain information...')
	domain = np.load(meta.file_domain)
	samples = np.load(meta.file_samples)
	noise_domain = np.load(meta.file_noise_domain)
except IOError:
	print('Domain information does not exists... building from dataset.')
	domain, samples, noise_domain = build_data(meta.file_data)
	np.save(meta.file_domain, domain)
	np.save(meta.file_samples, samples)
	np.save(meta.file_noise_domain, noise_domain)

domain_size = len(domain)
noise_domain_size = len(noise_domain)

print('{} samples available.'.format(len(samples)))

def xavier_variable(name, shape):
	if len(shape) == 1:
		inout = np.sum(shape) + 1
	else:
		inout = np.sum(shape)

	init_range = np.sqrt(6.0 / inout)
	initializer = tf.random_uniform_initializer(-init_range, init_range)
	return tf.get_variable(name, shape, initializer=initializer)

Q = xavier_variable('Q', shape=[domain_size, meta.embed_dim])
R = xavier_variable('R', shape=[domain_size, meta.embed_dim])
B = xavier_variable('B', shape=[domain_size])

w_data = tf.placeholder(tf.int32, shape=[meta.batch_size])
c_data = tf.placeholder(tf.int32, shape=[meta.batch_size])
w_noise = tf.placeholder(tf.int32, shape=[meta.batch_size * meta.noise_size])
c_noise = tf.placeholder(tf.int32, shape=[meta.batch_size * meta.noise_size])

q_w_data = tf.nn.embedding_lookup(Q, w_data)
r_c_data = tf.nn.embedding_lookup(R, c_data)
b_c_data = tf.nn.embedding_lookup(B, c_data)

q_w_noise = tf.nn.embedding_lookup(Q, w_noise)
r_c_noise = tf.nn.embedding_lookup(R, c_noise)
b_c_noise = tf.nn.embedding_lookup(B, c_noise)

score_data = tf.reduce_sum(tf.multiply(q_w_data, r_c_data), axis=1) + b_c_data
score_noise = -(tf.reduce_sum(tf.multiply(q_w_noise, r_c_noise), axis=1) + b_c_noise)

loss = -(tf.reduce_sum(tf.log(tf.sigmoid(score_data))) + tf.reduce_sum(tf.log(tf.sigmoid(score_noise))))
global_step = tf.Variable(0, trainable=False, name='global_step')
train = tf.train.GradientDescentOptimizer(meta.learning_rate).minimize(loss, global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	saver.restore(sess, ckpt.model_checkpoint_path)
else:
	sess.run(tf.global_variables_initializer())

def nearest(Q, key, k):
	embed = np.array([Q[key]])
	dot = np.sum(np.multiply(np.repeat(embed, Q.shape[0], axis=0), Q), axis=1)
	norm = np.linalg.norm(Q, axis=1) * np.linalg.norm(embed)
	cos = dot / norm
	return np.argpartition(-cos, k)[0:k]

for _ in range(meta.training_step):
	indices = np.random.choice(domain_size, size=[meta.batch_size])
	batch = samples[indices]
	feed = {
		w_data : batch[:,0],
		c_data : batch[:,1]
	}
	_, loss_value = sess.run([train, loss], feed_dict=feed)
	step = sess.run(global_step)

	if step % meta.interval_print == 0:
		print('{}: Loss {:g}'.format(step, loss_value))

	if step % meta.interval_save == 0:
		print('Saving status...')
		saver.save(sess, meta.savefile, global_step=global_step)

		Q_embed = sess.run(Q, feed_dict={})
		n = nearest(Q_embed, np.where(domain == 'one')[0][0], 5)
		print('one: {}'.format(domain[n]))    

		n = nearest(Q_embed, np.where(domain == 'good')[0][0], 5)
		print('good: {}'.format(domain[n])) 

		n = nearest(Q_embed, np.where(domain == 'water')[0][0], 5)
		print('water: {}'.format(domain[n])) 