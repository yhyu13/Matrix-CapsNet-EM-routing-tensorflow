from model.capsnet_model import CapsNet
from model.cnn_baseline import CNNBaseline

import tensorflow as tf
import numpy as np
import os
import sys

import logging
import daiquiri

from dataset import get_batch

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class Net(object):

    def __init__(self, flags, hps):

        self.num_classes = flags.n_classes
        self.img_row = flags.n_img_row
        self.img_col = flags.n_img_col
        self.img_channels = flags.n_img_channels
        self.num_batch = flags.n_batch
        self.load_model_path = flags.load_model_path

        tf.reset_default_graph()
        g = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.33
        self.sess = tf.Session(config=config, graph=g)

        with g.as_default():
            tf.set_random_seed(1234)
            self.imgs = tf.placeholder(tf.float32,
                                       shape=[self.num_batch if flags.MODE == 'train' else None,
                                              self.img_row, self.img_col, self.img_channels])
            self.labels = tf.placeholder(tf.float32,
                                         shape=[self.num_batch if flags.MODE == 'train' else None,
                                                self.num_classes])
            models = {'cap': lambda: CapsNet(hps, self.imgs, self.labels),
                      'cnn': lambda: CNNBaseline(hps, self.imgs, self.labels)}
            self.model = models[flags.model]()
            logger.debug("Building Model...")

            self.model.build_graph()
            var_to_save = tf.trainable_variables() + [var for var in tf.global_variables()
                                                      if ('bn' in var.name) and ('Adam' not in var.name) and ('Momentum' not in var.name) or ('global_step' in var.name)]
            logger.debug(
                f'Building Model Complete...Total parameters: {self.model.total_parameters(var_list=var_to_save)}')

            self.summary = self.model.summaries
            self.train_writer = tf.summary.FileWriter("./train_log", self.sess.graph)
            self.test_writer = tf.summary.FileWriter("./test_log")
            self.saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)
            logger.debug(f'Build Summary & Saver complete')

            self.initialize()
            self.restore_model()

    def close(self):
        self.sess.close()
        logger.info(f'Network shutdown!')

    def initialize(self):
        init = (var.initializer for var in tf.global_variables())
        self.sess.run(list(init))
        logger.info('Done initializing variables')

    def restore_model(self):
        if self.load_model_path is not None:
            logger.debug('Loading Model...')
            try:
                ckpt = tf.train.get_checkpoint_state(check_point_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                logger.debug('Loading Model Succeeded...')
            except:
                logger.debug('Loading Model Failed')
                pass

    def save_model(self, name):
        self.saver.save(self.sess, f'./savedmodels/model-{name}.ckpt',
                        global_step=self.sess.run(self.model.global_step))

    def predict(self, features):
        pass

    def train(self, porportion=0.25):
        logger.info('Train model...')
        num_iter = int(60000 * porportion // self.num_batch) + 1
        logger.info(f'1 Epoch training iteration will be: {num_iter}')
        for i in range(num_iter):
            batch = get_batch(train=True, batch_size=self.num_batch)

            feed_dict = {self.model.images: batch['x'],
                         self.model.labels: batch['y'],
                         self.model.is_training: True}
            try:
                _, summary, l, acc, lrn_rate = \
                    self.sess.run([self.model.train_op, self.summary, self.model.cost,
                                   self.model.acc, self.model.lrn_rate], feed_dict=feed_dict)
            except KeyboardInterrupt:
                self.close()
                sys.exit()
            except tf.errors.InvalidArgumentError:
                continue
            else:
                global_step = self.sess.run(self.model.global_step)
                self.train_writer.add_summary(summary, global_step)
                self.sess.run(self.model.increase_global_step)
                if i % 2 == 0:
                    logger.debug(
                        f'Train step {i} | Loss: {l:.3f} | Accuracy: {acc:.3f} | Global step: {global_step} | Learning rate: {lrn_rate}')

    def test(self, porportion=0.25):
        logger.info('Evaluate model...')
        num_iter = int(1000 * porportion // self.num_batch) + 1
        logger.info(f'1 Epoch training iteration will be: {num_iter}')
        t_l, t_acc = 0, 0
        for i in range(num_iter):
            batch = get_batch(train=False, batch_size=self.num_batch)

            feed_dict = {self.model.images: batch['x'],
                         self.model.labels: batch['y'],
                         self.model.is_training: False}
            try:
                summary, l, acc = \
                    self.sess.run([self.summary, self.model.cost,
                                   self.model.acc], feed_dict=feed_dict)
            except KeyboardInterrupt:
                self.close()
                sys.exit()
            except tf.errors.InvalidArgumentError:
                continue
            else:
                global_step = self.sess.run(self.model.global_step)
                self.test_writer.add_summary(summary, global_step)
                t_l += l
                t_acc += acc
                if i % 1 == 0:
                    logger.debug(
                        f'Test step {i} | Loss: {l:.3f} | Accuracy: {acc:.3f} | Global step: {global_step}')
        return t_l / num_iter, t_acc / num_iter
