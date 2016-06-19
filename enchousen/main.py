import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os
import time
from glob import glob

def discriminator(image, reuse=False):
    if reuse:
            tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv'))
    h1 = lrelu(d_bn1(conv2d(h0, 64, df_dim*2, name='d_h1_conv')))
    h2 = lrelu(d_bn2(conv2d(h1, 128, df_dim*4, name='d_h2_conv')))
    h3 = lrelu(d_bn3(conv2d(h2, 256, df_dim*8, name='d_h3_conv')))
    h4 = dense(tf.reshape(h3, [batchsize, -1]), 4*4*512, 1, scope='d_h3_lin')
    return tf.nn.sigmoid(h4), h4

def generator(z):
    z2 = dense(z, z_dim, gf_dim*8*4*4, scope='g_h0_lin')
    h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, gf_dim*8])))
    h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [batchsize, 8, 8, gf_dim*4], "g_h1")))
    h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [batchsize, 16, 16, gf_dim*2], "g_h2")))
    h3 = tf.nn.relu(g_bn3(conv_transpose(h2, [batchsize, 32, 32, gf_dim*1], "g_h3")))
    h4 = conv_transpose(h3, [batchsize, 64, 64, 3], "g_h4")
    return tf.nn.tanh(h4)

with tf.Session() as sess:
    batchsize = 64
    iscrop = True
    imagesize = 108
    imageshape = [64, 64, 3]
    z_dim = 100
    gf_dim = 64
    df_dim = 64
    gfc_dim = 1024
    dfc_dim = 1024
    c_dim = 3
    learningrate = 0.0002
    beta1 = 0.5
    dataset = "celebA"

    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')

    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')

    # build model
    images = tf.placeholder(tf.float32, [batchsize] + imageshape, name="real_images")
    zin = tf.placeholder(tf.float32, [None, z_dim], name="z")
    G = generator(zin)
    D_prob, D_logit = discriminator(images)

    D_fake_prob, D_fake_logit = discriminator(G, reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit, tf.ones_like(D_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.zeros_like(D_fake_logit)))

    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.ones_like(D_fake_logit)))
    dloss = d_loss_real + d_loss_fake

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    data = glob(os.path.join("./data", dataset, "*.jpg"))

    d_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(dloss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(gloss, var_list=g_vars)
    tf.initialize_all_variables().run()

    counter = 1
    start_time = time.time()
    for epoch in xrange(10):
        data = glob(os.path.join("./data",dataset, "*.jpg"))
        batch_idx = len(data)
        for idx in xrange(batch_idx):
            batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
            batch = [get_image(batch_file, imagesize, is_crop=iscrop) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)

            batch_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)
            sess.run([d_optim],feed_dict={ self.images: batch_images, self.z: batch_z })
            sess.run([g_optim],feed_dict={ self.z: batch_z })
            sess.run([g_optim],feed_dict={ self.z: batch_z })

            errD_fake = self.d_loss_fake.eval({self.z: batch_z})
            errD_real = self.d_loss_real.eval({self.images: batch_images})
            errG = self.g_loss.eval({self.z: batch_z})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake+errD_real, errG))
