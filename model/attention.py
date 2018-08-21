import os

import numpy as np
import tensorflow as tf

from model.model import Model
from utils.data_utils import read_vocab
from utils.model_utils import get_optimizer, load_pretrained_emb_from_txt


class AttentionModel(Model):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.name = config.model

    def build(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.setup_placeholders()
        self.setup_embedding()
        self.setup_attention()
        self.setup_fnn()
        if self.config.mode == 'train':
            self.setup_train()
        
        self.saver = tf.train.Saver(tf.global_variables())

    def setup_placeholders(self):
        # [batch_size]
        self.input_x = tf.placeholder(tf.int32, [None], name = 'X')
        # batch_size * window_size
        self.context_x = tf.placeholder(tf.int32, [None, None], name = "Context_X")
        # batch_size * num_classes
        self.input_y = tf.placeholder(tf.float32, [None, None], name='y')
        # dropout keep prob
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.l2_loss = tf.constant(0.0)

    def setup_embedding(self):
        with tf.variable_scope("Embedding"), tf.device("/cpu:0"):
            self.word2id, self.id2word = read_vocab(self.config.word_vocab_file)
            embedding = load_pretrained_emb_from_txt(self.id2word, self.config.pretrained_embedding_file)
            # vocab_size * embedding_size
            self.source_embedding = tf.get_variable("source_embedding", dtype=tf.float32, initializer=tf.constant(embedding), trainable=False)
            # embedding_size * W_dim
            self.W_w = tf.get_variable(
                "W_w",
                shape=[self.config.embedding_size, self.config.W_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            # vocab_size * W_dim
            self.hidden_embedding = tf.tanh(tf.matmul(self.source_embedding, self.W_w))
            # batch_size * embedding_size
            self.source_inputs = tf.nn.embedding_lookup(self.source_embedding, self.input_x)
            # batch_size * W_dim
            self.hidden_inputs = tf.nn.embedding_lookup(self.hidden_embedding, self.input_x)
            # batch_size * window_size * embedding_size
            self.source_context = tf.nn.embedding_lookup(self.source_embedding, self.context_x)
            # batch_size * window_size * W_dim
            self.hidden_context = tf.nn.embedding_lookup(self.hidden_embedding, self.context_x)

    def setup_attention(self):
        with tf.variable_scope("Attention"):
            # batch_size * 1 * W_dim
            self.hidden_inputs_expand = tf.expand_dims(self.hidden_inputs, 1)
            # batch_size * W_dim * window_size
            self.hidden_context_T = tf.transpose(self.hidden_context, perm=[0,2,1])
            # batch_size * 1 * window_size
            # this can be replaced by "tf.softmax(tf.matmul(a,b))"
            self.a = tf.nn.softmax(tf.einsum('aij,ajk->aik', self.hidden_inputs_expand, self.hidden_context_T))
            # batch_size * embedding_size
            self.c = tf.squeeze(tf.einsum('aij,ajk->aik', self.a, self.source_context))
            # batch_size * (2 * embedding_size)
            self.h = tf.concat([self.source_inputs, self.c], -1)
    
    def setup_fnn(self):
        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h, self.dropout_keep_prob)
        
        # three layer nn
        with tf.name_scope("output"):
            # (2 * embedding_size) * hidden_layer_size
            W_hidden = tf.get_variable(
                "hidden_layer_weights",
                shape = [2 * self.config.embedding_size, self.config.hidden_layer_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            # [hidden_layer_size]
            b_hidden = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_layer_size]), name = 'hidden_bias')
            # batch_size * hidden_layer_size
            self.hidden_layer = tf.nn.tanh(tf.nn.xw_plus_b(self.h_drop, W_hidden, b_hidden))

            self.l2_loss += tf.nn.l2_loss(W_hidden)
            self.l2_loss += tf.nn.l2_loss(b_hidden)

            # hidden_layer_size * num_classes
            W_output = tf.get_variable(
                "output_layer_weights",
                shape = [self.config.hidden_layer_size, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            # [num_classes]
            b_output = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name = "output_bias")
            # batch_size * num_classes
            self.scores = tf.nn.xw_plus_b(self.hidden_layer, W_output, b_output, name = "scores")
            # batch_size * num_classes
            self.softmax = tf.nn.softmax(self.scores)

            self.l2_loss += tf.nn.l2_loss(W_output)
            self.l2_loss += tf.nn.l2_loss(b_output)

            # [batch_size]
            self.pred = tf.argmax(self.scores, 1, name = "predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name = "accuracy")

    def train_one_step(self, x_batch, context_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.context_x: context_batch, 
            self.input_y: y_batch,
            self.dropout_keep_prob: self.config.dropout_keep_prob
        }

        loss, accuracy, global_step, _ = self.sess.run([self.loss, self.accuracy, self.global_step, self.updates], feed_dict = feed_dict)
        return loss, accuracy, global_step

    def inference(self, sen, non_event_id=None):
        x_batch, context_batch = zip(*sen)
        feed_dict = {
            self.input_x: x_batch,
            self.context_x: context_batch,
            self.dropout_keep_prob: 1.0
        } 
        if non_event_id == None:
            tag2id, _ = read_vocab(self.config.tag_vocab_file)
            non_event_id = tag2id['__label__非事件']
        prob = self.sess.run(self.softmax, feed_dict=feed_dict)
        prob_max = np.max(prob, axis = 1).tolist()
        prob_idx = np.argmax(prob, axis = 1).tolist()
        max_prob = 0
        event_type_id = non_event_id
        for i in range(len(prob_idx)):
            if prob_idx[i] == non_event_id:
                continue
            if prob_max[i] > max_prob:
                max_prob = prob_max[i]
                event_type_id = prob_idx[i]
        return [event_type_id]

    def evaluate(self, sen, non_event_id):
        #TODO: if non_event_id == None, read from config.tag_vocab_file
        x_batch, context_batch, label = zip(*sen)

        pre_event_id = self.inference(zip(x_batch, context_batch), non_event_id)[0]

        if isinstance(label[0], int):
            true_label = set(label)
        else:
            true_label = set(np.argmax(label, 1).tolist())
        true_label.remove(non_event_id)
        if len(true_label) == 0:
            true_event_id = non_event_id
        else:
            true_event_id = true_label.pop()
        # print(pre_event_id, true_event_id)
        if pre_event_id == true_event_id:
            return True
        return False

