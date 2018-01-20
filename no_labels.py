import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
from next_batch_partial import next_batch_partial

class LatentAttention():
    def __init__(self, frac_train, n_z, batchsize):
        """
        frac_train: (0..1) the fraction of the training set to use for
            training ... the rest will be used for validation
        n_z: (positive int) number of latent gaussian variables consumed by
            the decoder / produced by the endcoder
        batchize: (positive int) number of items to include in each training
            minibatch
        """
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_train = int(frac_train * self.mnist.train.num_examples)
        self.n_test = self.mnist.train.num_examples - self.n_train

        assert batchsize <= self.n_test

        self.n_z = n_z
        self.batchsize = batchsize

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal(z_stddev.get_shape(),0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generate_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generate_images, [-1, 28*28])

        self.calc_generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.calc_latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.calc_generation_loss + self.calc_latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[-1, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [-1, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def print_epoch(self, epoch, gen_loss, lat_loss, saver, sess,
                    visualization):
        print("epoch {}: genloss {} latloss {}".format(
            epoch,
            np.mean(gen_loss), np.mean(lat_loss)))

        saver.save(sess, os.getcwd()+"/training/train",
                   global_step=epoch)
        generated_test = sess.run(
            self.generate_images,
            feed_dict={self.images: visualization}).reshape(
                -1, 28, 28)
        ims("results/"+str(epoch)+".jpg",
            merge(generated_test[:64], [8, 8]))

    def train(self):
        data = self.mnist.train
        vis_size = self.batchsize  # Recognition assumes entries are
                                   # this long ... so must only use
                                   # this much for the visualization
        if self.n_test == 0:
            visualization, vis_labels = next_batch_partial(
                data, vis_size, self.n_train)
        else:
            visualization = data.images[self.n_train:self.n_train+vis_size]
            vis_labels = data.labels[self.n_train:self.n_train+vis_size]

        reshaped_vis = visualization.reshape(vis_size,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            last_epochs_completed = -1
            while(data.epochs_completed < 10):
                batch, batch_labels = next_batch_partial(
                    data, self.batchsize, self.n_train)
                _, gen_loss, lat_loss = sess.run(
                    (self.optimizer, self.calc_generation_loss,
                     self.calc_latent_loss),
                    feed_dict={self.images: batch})
                if last_epochs_completed != data.epochs_completed:
                    last_epochs_completed = data.epochs_completed
                    self.print_epoch(
                        last_epochs_completed, gen_loss, lat_loss,
                        saver, sess, visualization
                    )


if __name__ == '__main__':
    model = LatentAttention(0.9, 20, 100)
    model.train()
