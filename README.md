word-embedding-tensorflow
=============
# Introduction

Word Embedding Solver using Tensorflow

# Dependencies

	Python 3.5.3
	Tensorflow-GPU 1.2.1

# How to use

	python embed.py

It will automatically generate dataset from given text file. The original dataset can be generated from [this][data].

Also, it will test its accuracy every 10000 steps. The test file can be downloaded [here][test] in word2vec/trunk/questions-words.txt.

# Reference

The original paper is <[Learning word embeddings efficiently with noise-contrastive estimation][paper]>. Check out the paper for the theoretical details.

[paper]: http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
[data]: http://mattmahoney.net/dc/text8.zip
[test]: https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip