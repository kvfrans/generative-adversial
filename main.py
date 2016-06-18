import tensorflow as tf
import numpy as np
import cPickle
from random import randint
import os
from scipy.misc import imsave
import utils



batchsize = 50
droprate = 0.9
iterations = 10000
updates = 2
d_learnrate = 0.0002
g_learnrate = 0.0002
beta = 0.5


# extracting data
def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

# helper functions for defining the model
def initWeight(shape, name):
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.05))
# start with 0.1 so reLu isnt always 0
def initBias(shape, name):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))


def discriminator(imagein):
    d_convweight1 = initWeight([5,5,3,32],"d_convweight1")
    d_convweight2 = initWeight([5,5,32,64],"d_convweight2")
    d_convweight3 = initWeight([5,5,64,128],"d_convweight3")

    d_convbias1 = initBias([32],"d_convbias1")
    d_convbias2 = initBias([64],"d_convbias2")
    d_convbias3 = initBias([128],"d_convbias3")

    d_fcweight1 = initWeight([8*8*128, 1],"d_fcweight1")

    d_fcbias1 = initBias([1],"d_fcbias1")

    conv1 = tf.nn.conv2d(imagein,d_convweight1,strides=[1,1,1,1],padding="SAME")
    relu1 = tf.nn.relu(conv1 + d_convbias1)
    conv2 = tf.nn.conv2d(relu1,d_convweight2,strides=[1,2,2,1],padding="SAME") #32x32 -> 16x16
    relu2 = tf.nn.relu(conv2 + d_convbias2)
    conv3 = tf.nn.conv2d(relu2,d_convweight3,strides=[1,2,2,1],padding="SAME") #16x16 -> 8x8
    relu3 = tf.nn.relu(conv3 + d_convbias3)
    dropout1 = tf.nn.dropout(relu3,droprate)
    flattened = tf.reshape(dropout1, [-1, 8*8*128])
    fc1 = tf.matmul(flattened, d_fcweight1) + d_fcbias1
    y = tf.nn.sigmoid(fc1)
    return y, fc1

def generator():
    g_fcweight1 = initWeight([100, 4*4*512],"g_fcweight1")
    g_fcbias1 = initBias([4*4*512],"g_fcbias1")

    g_convweight1 = initWeight([5,5,256,512],"g_convweight1")
    g_convweight2 = initWeight([5,5,128,256],"g_convweight2")
    g_convweight3 = initWeight([5,5,3,128],"g_convweight3")

    g_convbias1 = initBias([256],"g_convbias1")
    g_convbias2 = initBias([128],"g_convbias2")
    g_convbias3 = initBias([3],"g_convbias3")

    noise = tf.random_uniform([batchsize, 100], -1, 1)
    fc1 = tf.matmul(noise, g_fcweight1)
    relu1 = tf.nn.relu(fc1 + g_fcbias1)
    fattened = tf.reshape(relu1,[batchsize, 4, 4, 512])
    convt1 = tf.nn.conv2d_transpose(fattened, g_convweight1, [batchsize, 8, 8, 256], [1,2,2,1])
    relu2 = tf.nn.relu(convt1 + g_convbias1)
    convt2 = tf.nn.conv2d_transpose(relu2, g_convweight2, [batchsize, 16, 16, 128], [1,2,2,1])
    relu3 = tf.nn.relu(convt2 + g_convbias2)
    convt3 = tf.nn.conv2d_transpose(relu3, g_convweight3, [batchsize, 32, 32, 3], [1,2,2,1])
    sigmoid1 = tf.nn.sigmoid(convt3 + g_convbias3)
    return sigmoid1;




def train(mode):

    # prepare real data
    batch = unpickle("cifar-10-batches-py/data_batch_1")


    imagein = tf.placeholder("float",[batchsize,32,32,3])

    with tf.variable_scope("model"):
        # discriminator called on real data
        preal, hreal = discriminator(imagein)
        # discriminator called on generated images
        g = generator()
    # want to reuse same discriminator variables
    with tf.variable_scope("model", reuse=True):
        pfake, hfake = discriminator(g)

    # parameters to train on for each respective update
    params = tf.trainable_variables()
    d_params = [var for var in params if 'd_' in var.name]
    g_params = [var for var in params if 'g_' in var.name]
    print "params are"
    print [var.name for var in d_params]

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hreal, tf.ones_like(hreal)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hfake, tf.zeros_like(hfake)))
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hfake, tf.ones_like(hfake)))
    dloss = d_loss_real + d_loss_fake


    # d_loss_real2 = tf.reduce_mean(tf.nn.relu(preal) - preal + tf.log(1.0 + tf.exp(-tf.abs(preal))))
    # d_loss_fake2 = tf.reduce_mean(tf.nn.relu(pfake) + tf.log(1.0 + tf.exp(-tf.abs(pfake))))
    # gloss = tf.reduce_mean(tf.nn.relu(d2) - d2 + tf.log(1.0 + tf.exp(-tf.abs(d2))))

    doptimizer = tf.train.AdamOptimizer(d_learnrate, beta1=beta).minimize(dloss, var_list=d_params)
    goptimizer = tf.train.AdamOptimizer(g_learnrate, beta1=beta).minimize(gloss, var_list=g_params)

    saver = tf.train.Saver(max_to_keep=10)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        if mode == "train":
            for i in xrange(iterations):
                for k in xrange(updates):
                    randomint = randint(0,10000 - batchsize - 1)
                    trainingData = batch["data"][randomint:batchsize+randomint]
                    rawlabel = batch["labels"][randomint:batchsize+randomint]
                    trainingLabel = np.zeros((batchsize,10))
                    trainingLabel[np.arange(batchsize),rawlabel] = 1
                    trainingData = trainingData/255.0
                    trainingData = np.reshape(trainingData,(batchsize,32,32,3))

                    doptimizer.run(feed_dict= {imagein: trainingData})

                    if i % 50 == 0:
                        dd = dloss.eval(feed_dict= {imagein: trainingData})
                        print dd
                    # dloss += dloss_delta
                    if k % 100 == 0:
                        print i

                goptimizer.run()
                # print "i: " + str(i)



                if i % 100 == 0:
                    saver.save(sess, os.getcwd()+"/training/train", global_step=i)
                    data = g.eval()
                    imsave(str(i)+".jpg",data[0])
                # _, gloss_delta = goptimizer.run()
                # gloss += gloss_delta
        else:
            saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
            data = g.eval()
            imsave(str(i)+".jpg",data)



#
# def display():
#     sess = tf.InteractiveSession()
#     sess.run(tf.initialize_all_variables())
#     g = generator()
#     d2 = discriminator(g)
#     saver = tf.train.Saver()
#     saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))




# display()

train("train")
