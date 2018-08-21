import math

import tensorflow as tf

from model.model import Model
from utils.data_utils import read_vocab
from utils.model_utils import get_optimizer, load_pretrained_emb_from_txt
import time


class SelfAttention(Model):
    """
    A self-attention model for text classification.
    attention module is the encoder of transformer model, then followed by a fnn layer.
    """
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.name = config.model

    def build(self):
        print("build graph")
        self.global_step = tf.Variable(0, trainable=False)
        self.setup_placeholders()
        self.setup_embedding()
        self.setup_self_attention()
        self.setup_fnn()
        if self.config.mode == 'train':
            self.setup_train()
        self.saver = tf.train.Saver(tf.global_variables())

    def setup_placeholders(self):
        # batch_size * sentence_length
        self.input_x = tf.placeholder(tf.int32, [None, None], name='X')
        # batch_size * num_classes
        self.input_y = tf.placeholder(tf.int32, [None, None], name='y')
        # dropout keep prob
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')
        # L2 Regularization (opt.)
        # self.l2_reg_lambda = tf.placeholder(tf.float32, name='L2_Regularization')
        # self.l2_loss = tf.constant(0.0)

    def setup_embedding(self):
        with tf.variable_scope("Embedding"), tf.device("/cpu:0"):
            self.word2id, self.id2word = read_vocab(self.config.word_vocab_file)
            start_time = time.time()
            embedding = load_pretrained_emb_from_txt(self.id2word, self.config.pretrained_embedding_file)
            # start_time = time.time()
            self.source_embedding = tf.get_variable("source_emebdding", dtype=tf.float32, initializer=tf.constant(embedding), trainable=False)
            end_time = time.time()
            print(end_time - start_time)
            # batch_size * sentence_length * embedding_size
            self.source_inputs = tf.nn.embedding_lookup(self.source_embedding, self.input_x)
        
    def attention_fun(self, q, k, v):
        score = tf.einsum('ij,ajk->aik', q, k)
        a = tf.nn.softmax(tf.div(score, math.sqrt(self.config.embedding_size)))
        
        c = tf.squeeze(tf.matmul(a, v), axis=[1])
        return c
        
    def setup_self_attention(self):
        with tf.variable_scope("attention"):
            # attention_outputs = []
            # for i in range(self.config.num_query):
                # with tf.name_scope("attention-%s" % i):
                    # 1 * 1 * embedding_size
                    # query = tf.get_variable(
                    #     "Query",
                    #     shape=[1,self.config.embedding_size],
                    #     initializer=tf.contrib.layers.xavier_initializer())
            self.query = tf.Variable(tf.truncated_normal([1,self.config.embedding_size], stddev=0.1), name='query')
            # batch_size * sentence_length * embedding_size
            key = self.source_inputs
            # batch_size * embedding_size * sentence_length
            k_T = tf.transpose(key, perm=[0,2,1])
            # batch_size * sentence_length * embedding_size
            value = self.source_inputs
            # batch_size * embedding_size
            self.h1 = self.attention_fun(self.query, k_T, value)
                    # attention_outputs.append(h)
            # self.h = tf.reduce_mean(attention_outputs, axis=0)
            # embedding_size * 100
            W_attention = tf.get_variable(
                "W_attention", 
                shape = [self.config.embedding_size, 150],
                initializer=tf.contrib.layers.xavier_initializer())
            # batch_size * 100
            query2 = tf.matmul(self.h1, W_attention)
            # batch_size * 1 * 100
            query2 = tf.expand_dims(query2, 1)
            # batch_size * sentence_length * 100
            key2 = tf.einsum('aij,jk->aik', key, W_attention)
            # batch_size * 100 * sentence_length
            key2_T = tf.transpose(key2, perm=[0,2,1])
            # batch_size * 1 * sentence_length
            a2 = tf.nn.softmax(tf.matmul(query2, key2_T))
            # batch_size * 1 * embedding_size          
            self.h2 = tf.squeeze(tf.matmul(a2, value), axis=[1])
            # batch_size * (2*embeddingsize)
            self.h = tf.concat([self.h1, self.h2], axis=-1)

            

    def setup_fnn(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h, self.dropout_keep_prob)

        with tf.name_scope("output"):
            # embedding_size * num_classes
            W = tf.get_variable(
                "W",
                shape = [2*self.config.embedding_size, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # [num_classes]
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
            # batch_size * num_classes
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
            # [batch_size]
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name = "accuracy")

    def test(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.config.dropout_keep_prob
        }
        k_T_shape, q_shape = self.sess.run([self.k_T_shape, self.q_shape], feed_dict = feed_dict)
        return k_T_shape, q_shape

    def train_one_step(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.config.dropout_keep_prob
        }
        loss, accuracy, global_step, _ = self.sess.run([self.loss, self.accuracy, self.global_step, self.updates], feed_dict=feed_dict)
        return loss, accuracy, global_step
            
    def evaluate(self, x_batch, y_batch):
        """evaluate model with eval dataset"""
        feed_dict = {
            self.input_x: x_batch, 
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0,
        }
        accuracy = self.sess.run([self.accuracy], feed_dict=feed_dict)
        return accuracy

    def inference(self, input_x):
        feed_dict = {
            self.input_x: input_x,
            self.dropout_keep_prob: 1.0
        }
        pred = self.sess.run(self.predictions, feed_dict = feed_dict)
        return pred
