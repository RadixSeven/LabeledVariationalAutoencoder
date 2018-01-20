import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
from scipy.misc import imsave as ims
from utils import merge
from ops import conv2d, conv_transpose, dense, lrelu
from next_batch_partial import next_batch_partial


class LatentAttention():
    def __init__(self, frac_train, n_z, batchsize, learning_rate,
                 e_h1, e_h2, d_h1, d_h2):
        """
        frac_train: (0..1) the fraction of the training set to use for
            training ... the rest will be used for validation
        n_z: (positive int) number of latent gaussian variables consumed by
            the decoder / produced by the endcoder
        batchize: (positive int) number of items to include in each training
            minibatch
        learning_rate: (positive float) the learning rate used by the
            optimizer
        e_h1: (positive integer) number of channels in output of first hidden
            layer in encoder
        e_h2: (positive integer) number of layers in output of second hidden
            layer in encoder
        d_h1: (positive integer) number of channels in input of first hidden
            layer in decoder
        d_h2: (positive integer) number of layers in input of second hidden
            layer in decoder
        """
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_train = int(frac_train * self.mnist.train.num_examples)
        self.n_test = self.mnist.train.num_examples - self.n_train
        self.e_h1 = e_h1
        self.e_h2 = e_h2
        self.d_h1 = d_h1
        self.d_h2 = d_h2

        assert batchsize <= self.n_test

        self.n_z = n_z
        self.batchsize = batchsize

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])
        z_mean, z_stddev = self.encode(image_matrix)
        samples = tf.random_normal(tf.shape(z_stddev), 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generate_images = self.decode(guessed_z)
        generated_flat = tf.reshape(self.generate_images, [-1, 28*28])

        self.calc_generation_loss = -tf.reduce_sum(
            self.images * tf.log(1e-8 + generated_flat)
            + (1-self.images) * tf.log(1e-8 + 1 - generated_flat), 1)

        self.calc_latent_loss = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_stddev) -
            tf.log(tf.square(z_stddev)) - 1, 1)
        self.cost = tf.reduce_mean(
            self.calc_generation_loss + self.calc_latent_loss)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(self.cost)

    def encode(self, input_images):
        with tf.variable_scope("encode"):
            # 28x28x1 -> 14x14x16
            h1 = lrelu(conv2d(input_images, 1, self.e_h1, "e_h1"))
            # 14x14x16 -> 7x7x32
            h2 = lrelu(conv2d(h1, self.e_h1, self.e_h2, "e_h2"))
            h2_flat = tf.reshape(h2, [-1, 7*7*self.e_h2])

            w_mean = dense(h2_flat, 7*7*self.e_h2, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*self.e_h2, self.n_z, "w_stddev")

        return w_mean, w_stddev

    def decode(self, z):
        with tf.variable_scope("decode"):
            z_shape = tf.shape(z)
            z_develop = dense(z, self.n_z, 7*7*self.d_h1, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [-1, 7, 7, self.d_h1]))
            h1 = tf.nn.relu(conv_transpose(
                z_matrix, [z_shape[0], 14, 14, self.d_h2], "d_h1"))
            h2 = conv_transpose(h1, [z_shape[0], 28, 28, 1], "d_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def print_epoch(self, epoch, gen_loss, lat_loss, saver, sess,
                    validation):

        saver.save(sess, os.getcwd()+"/training/train",
                   global_step=epoch)
        val_ims, val_error = sess.run(
            [self.generate_images, self.calc_generation_loss],
            feed_dict={self.images: validation})
        ims("results/"+str(epoch)+".jpg",
            merge(val_ims.reshape(-1, 28, 28)[:64], [8, 8]))

        self.val_error = np.mean(val_error)
        print("epoch {:02d}: genloss {:7.3f} latloss {:7.3f} "
              "validation_genloss {:7.3f}".format(
                  epoch,
                  np.mean(gen_loss), np.mean(lat_loss), self.val_error))

    def train(self):
        data = self.mnist.train
        if self.n_test == 0:
            validation, val_labels = next_batch_partial(
                data, self.batchsize, self.n_train)
        else:
            validation = data.images[self.n_train:]
            val_labels = data.labels[self.n_train:]

        reshaped_val = validation.reshape(-1, 28, 28)
        ims("results/base.jpg", merge(reshaped_val[:64], [8, 8]))
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
                        saver, sess, validation
                    )


if __name__ == '__main__':
    model = LatentAttention(
        0.9, 20, 100, 0.001,
        16, 32, 32, 16
    )
    model.train()
