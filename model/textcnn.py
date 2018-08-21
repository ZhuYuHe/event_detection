import os

import numpy as np
import tensorflow as tf

from model.model import Model
from utils.data_utils import read_vocab
from utils.model_utils import get_optimizer, load_pretrained_emb_from_txt


class TextCNN(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.name = config.model

    def build(self):
        print('build graph')
        self.global_step = tf.Variable(0, trainable=False)
        self.setup_placeholders()
        self.setup_embedding()
        self.setup_CNNs()
        self.setup_fnn()
        if self.config.mode == 'train':
            self.setup_train()
        
        self.saver = tf.train.Saver(tf.global_variables())

    def setup_placeholders(self):
        # batch_size * sentence_length
        self.input_x = tf.placeholder(tf.int32, [None, None], name='X')
        # batch_size * num_classes
        self.input_y = tf.placeholder(tf.float32, [None, None], name='y')
        # dropout keep prob
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # L2 regularization 
        self.l2_reg_lambda = tf.placeholder(tf.float32, name='L2_regularization_coef')
        # batch - sentence length
        # self.sentence_length = tf.placeholder(tf.int32, name="sentence_length")

        self.l2_loss = tf.constant(0.0)

    def setup_embedding(self):
        with tf.variable_scope("Embedding"), tf.device("/cpu:0"):
            self.word2id, self.id2word = read_vocab(self.config.word_vocab_file)
            embedding = load_pretrained_emb_from_txt(self.id2word, self.config.pretrained_embedding_file)
            self.source_embedding = tf.get_variable("source_emebdding", dtype=tf.float32, initializer=tf.constant(embedding), trainable=False)
            # batch_size * sentence_length * embedding_size
            self.source_inputs = tf.nn.embedding_lookup(self.source_embedding, self.input_x)
            # batch_size * sentence_length * embedding_size * 1
            self.source_inputs_expand = tf.expand_dims(self.source_inputs, -1)

    def setup_CNNs(self):
        with tf.variable_scope("Conv_Pool"):
            pooled_outputs = []
            # 按照filter_size卷积
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                    # W: filter_size * embedding_size * 1 * num_filters: 使用 num_filters 个 filter_size * embedding_size * 1 大小的 filter 进行卷积
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                    # self.source_inputs_expand: batch_size * sentence_length * embedding_size * 1
                    # W:                         filter_size * embedding_size * 1 * num_filters
                    # conv:                      batch_size * ( sentence_length - filter_size + 1 ) * 1 * num_filters
                    conv = tf.nn.conv2d(
                        self.source_inputs_expand,
                        W,
                        strides = [1,1,1,1],
                        padding="VALID",
                        name='conv'
                    )
                    # h: batch_size * ( sentence_length - filter_size + 1 ) * 1 * num_filters
                    h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name = 'relu')
                    # pooled: batch_size * 1 * 1 * num_filters
                    pooled = tf.nn.max_pool(
                        h,
                        # ksize 的参数需要是int，但是self.sentence_length是个node, 后面进行加减操作, 整个表达式变成了一个add node
                        # 后面将所有的句子处理成同一长度，使sentence_length为一常数
                        ksize=[1, self.config.sentence_length - filter_size + 1, 1, 1],
                        strides=[1,1,1,1],
                        padding = 'VALID',
                        name='pool'
                    )
                    pooled_outputs.append(pooled)
        # Combine all the pooled features
        self.num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        # self.h_pool: batch_size * 1 * 1 * num_filters_total
        self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool_flat: batch_size * num_filters_total
        # -1 can be used to inference shape. 
        # for example, if t has shape of [3,2,3], -1 in [2, -1, 3] refers to 3, -1 in [-1, 9] refers to 2;
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])


    def setup_fnn(self):
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        # Final scores and predictions
        with tf.name_scope("output"):
            # W: num_filters_total * num_classes
            W = tf.get_variable(
                "W", 
                shape = [self.num_filters_total, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # b: [num_classes]
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            # self.scores: batch_size * num_classes
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # self.predictions: [batch_size]
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # Calculate loss 
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def train_one_step(self, x_batch, y_batch):
        """A single train step. eg. a batch"""
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.config.dropout_keep_prob,
            self.l2_reg_lambda: self.config.l2_reg_lambda,
        }
        loss, accuracy, global_step, _ = self.sess.run([self.loss, self.accuracy, self.global_step, self.updates], feed_dict=feed_dict)
        return loss, accuracy, global_step

    def evaluate(self, x_batch, y_batch):
        """evaluate model with eval dataset"""
        feed_dict = {
            self.input_x: x_batch, 
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0,
            self.l2_reg_lambda: self.config.l2_reg_lambda
        }
        accuracy = self.sess.run([self.accuracy], feed_dict=feed_dict)
        return accuracy

    def inference(self, input_x):
        """
        predict label given a sentence
        """
        feed_dict = {
            self.input_x: input_x, 
            self.dropout_keep_prob: 1.0
        }

        prediction = self.sess.run(self.predictions, feed_dict=feed_dict)
        return prediction




