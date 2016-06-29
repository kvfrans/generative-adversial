import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os
import time
from glob import glob
from scipy.misc import imsave as ims
from random import randint

cifar = False

def discriminator(image, reuse=False):
    if reuse:
            tf.get_variable_scope().reuse_variables()

    if cifar:
        h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv')) #16x16x64
        h1 = lrelu(d_bn1(conv2d(h0, df_dim, df_dim*2, name='d_h1_conv'))) #8x8x128
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*2, df_dim*4, name='d_h2_conv'))) #4x4x256
        h4 = dense(tf.reshape(h2, [batchsize, -1]), 4*4*df_dim*4, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4), h4
    else:
        h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, 64, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, 128, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, 256, df_dim*8, name='d_h3_conv')))
        h4 = dense(tf.reshape(h3, [batchsize, -1]), 4*4*512, 1, scope='d_h3_lin')
        return tf.nn.sigmoid(h4), h4

def generator(z):
    if cifar:
        z2 = dense(z, z_dim, 4*4*gf_dim*4, scope='g_h0_lin')
        h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, gf_dim*4]))) # 4x4x256
        h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [batchsize, 8, 8, gf_dim*2], "g_h1"))) #8x8x128
        h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [batchsize, 16, 16, gf_dim*1], "g_h2"))) #16x16x64
        h4 = conv_transpose(h2, [batchsize, 32, 32, 3], "g_h4")
        return tf.nn.tanh(h4)
    else:
        z2 = dense(z, z_dim, gf_dim*8*4*4, scope='g_h0_lin')
        h0 = tf.nn.relu(g_bn0(tf.reshape(z2, [-1, 4, 4, gf_dim*8])))
        h1 = tf.nn.relu(g_bn1(conv_transpose(h0, [batchsize, 8, 8, gf_dim*4], "g_h1")))
        h2 = tf.nn.relu(g_bn2(conv_transpose(h1, [batchsize, 16, 16, gf_dim*2], "g_h2")))
        h3 = tf.nn.relu(g_bn3(conv_transpose(h2, [batchsize, 32, 32, gf_dim*1], "g_h3")))
        h4 = conv_transpose(h3, [batchsize, 64, 64, 3], "g_h4")
        return tf.nn.tanh(h4)

with tf.Session() as sess:
    batchsize = 64
    iscrop = False
    imagesize = 108
    imageshape = [64, 64, 3]
    if cifar:
        imageshape = [32, 32, 3]
    z_dim = 100
    gf_dim = 64
    df_dim = 64
    if cifar:
        gf_dim = 32
        df_dim = 32
    gfc_dim = 1024
    dfc_dim = 1024
    c_dim = 3
    learningrate = 0.0002
    beta1 = 0.5
    dataset = "imagenet"

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

    data = None
    batch = None
    if cifar:
        batch = unpickle("cifar-10-batches-py/data_batch_1")
    else:
        data = glob(os.path.join("./data", dataset, "*.JPEG"))
        print len(data)

    d_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(dloss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learningrate, beta1=beta1).minimize(gloss, var_list=g_vars)
    tf.initialize_all_variables().run()

    saver = tf.train.Saver(max_to_keep=10)

    counter = 1
    start_time = time.time()

    display_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)

    realfiles = data[0:64]
    realim = [get_image(batch_file, [64,64,3], is_crop=False) for batch_file in realfiles]
    real_img = np.array(realim).astype(np.float32)
    ims("results/imagenet/real.jpg",merge(real_img,[8,8]))

    train = True
    if train:
        # saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        for epoch in xrange(10):
            batch_idx = 30000 if cifar else (len(data)/batchsize)-2
            for idx in xrange(batch_idx):
                batch_images = None
                if cifar:
                    batchnum = randint(0,150)
                    trainingData = batch["data"][batchnum*batchsize:(batchnum+1)*batchsize]
                    trainingData = transform(trainingData, is_crop=False)
                    batch_images = np.reshape(trainingData,(batchsize,3,32,32))
                    batch_images = np.swapaxes(batch_images,1,3)
                else:
                    batch_files = data[idx*batchsize:(idx+1)*batchsize]
                    batchim = [get_image(batch_file, [64,64,3], is_crop=False) for batch_file in batch_files]
                    batch_images = np.array(batchim).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)

                for k in xrange(1):
                    sess.run([d_optim],feed_dict={ images: batch_images, zin: batch_z })
                for k in xrange(1):
                    sess.run([g_optim],feed_dict={ zin: batch_z })



                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, " \
                    % (epoch, idx, batch_idx,
                        time.time() - start_time,))

                if counter % 200 == 0:
                    sdata = sess.run([G],feed_dict={ zin: display_z })
                    print np.shape(sdata)
                    ims("results/imagenet/"+str(counter)+".jpg",merge(sdata[0],[8,8]))
                    errD_fake = d_loss_fake.eval({zin: display_z})
                    errD_real = d_loss_real.eval({images: batch_images})
                    errG = gloss.eval({zin: batch_z})
                    print errD_real + errD_fake
                    print errG
                    # print("errd: %4.4f errg: $4")
                if counter % 1000 == 0:
                    saver.save(sess, os.getcwd()+"/training/train",global_step=counter)
    else:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        batch_z = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
        batch_z = np.repeat(batch_z, batchsize, axis=0)
        for i in xrange(z_dim):
            edited = np.copy(batch_z)
            edited[:,i] = (np.arange(0.0, batchsize) / (batchsize/2)) - 1
            sdata = sess.run([G],feed_dict={ zin: edited })
            ims("results/imagenet/"+str(i)+".jpg",merge(sdata[0],[8,8]))
