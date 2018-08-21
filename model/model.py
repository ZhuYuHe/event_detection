import tensorflow as tf 
import os
from utils.model_utils import get_optimizer

class Model(object):
    def __init__(self):
        pass

    def build(self):
        raise NotImplementedError
    
    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def train_one_step(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def name(self):
        return self.name

    def setup_train(self):
        print("setting train...")
        self.learning_rate = tf.constant(self.config.learning_rate)
        # if self.config.learning_decay:
        #     if self.config.start_decay_step:
        #         start_decay_step = self.config.start_decay_step
        #     else:
        #         start_decay_step = int(self.config.num_train_steps / 2)
        #     remain_steps = self.config.num_train_steps - start_decay_step
        #     decay_steps = int(remain_steps / self.config.decay_times)
        #     print("learning rate - {0}, start decay step - {1}, decay ratio - {2}, decay times - {3} ".format(
        #         self.config.learning_rate, start_decay_step, self.config.decay_factor, self.config.decay_times
        #     ))
        #     self.learning_rate = tf.cond(
        #         self.global_step < start_decay_step,
        #         lambda: self.learning_rate,
        #         lambda: tf.train.exponential_decay(
        #             self.learning_rate,
        #             (self.global_step - start_decay_step),
        #             self.config.decay_factor, self.config.decay_factor, staircase=True), name = "learning_rate_decay_cond")
        opt = get_optimizer(self.config.optimizer)(self.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(zip(gradients, params), global_step=self.global_step)

    def save_model(self, checkpoint_dir=None, epoch=None):
        if epoch is None:
            self.saver.save(self.sess, os.path.join(self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir,
                        "model.ckpt"), global_step=self.global_step)
        else:
            self.saver.save(self.sess, os.path.join(self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir,
                        "model.ckpt"), global_step=epoch)

    def restore_model(self, checkpoint_dir=None, epoch=None):
        if epoch is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(
                self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir
            ))
        else:
            self.saver.restore(self.sess, os.path.join(self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir,
                               "model.ckpt" + ("-%d" % epoch)))

    def inference(self, input_x):
        """
        predict label given a sentence
        """
        raise NotImplementedError