import tensorflow as tf
import numpy as np
import cPickle
from random import randint



batchsize = 50
droprate = 0.9
iterations = 100
updates = 1000


# extracting data
def unpickle(file):
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

# helper functions for defining the model
def initWeight(shape):
    weights = tf.truncated_normal(shape,stddev=0.05)
    return tf.Variable(weights)
# start with 0.1 so reLu isnt always 0
def initBias(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)
# the convolution with padding of 1 on each side, and moves by 1.
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
# max pooling basically shrinks it by 2x, taking the highest value on each feature.
def maxPool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def discriminator(imagein):
    d_convweight1 = initWeight([5,5,3,32])
    d_convweight2 = initWeight([5,5,32,64])
    d_convweight3 = initWeight([5,5,64,128])

    d_convbias1 = initBias([32])
    d_convbias2 = initBias([64])
    d_convbias3 = initBias([128])

    d_fcweight1 = initWeight([8*8*128, 1024])
    d_fcweight2 = initWeight([1024, 1024])

    d_fcbias1 = initBias([1024])
    d_fcbias2 = initBias([1024])

    conv1 = tf.nn.conv2d(imagein,d_convweight1,strides=[1,1,1,1],padding="SAME")
    relu1 = tf.nn.relu(conv1 + d_convbias1)
    conv2 = tf.nn.conv2d(relu1,d_convweight2,strides=[1,2,2,1],padding="SAME") #32x32 -> 16x16
    relu2 = tf.nn.relu(conv2 + d_convbias2)
    conv3 = tf.nn.conv2d(relu2,d_convweight3,strides=[1,2,2,1],padding="SAME") #16x16 -> 8x8
    relu3 = tf.nn.relu(conv3 + d_convbias3)
    dropout1 = tf.nn.dropout(relu3,droprate)
    flattened = tf.reshape(dropout1, [-1, 8*8*128])
    fc1 = tf.matmul(flattened, d_fcweight1)
    relu4 = tf.nn.relu(fc1 + d_fcbias1)
    fc2 = tf.matmul(relu4, d_fcweight2)
    relu5 = tf.nn.relu(fc2 + d_fcbias2)
    return relu5;

def generator():
    g_fcweight1 = initWeight([100, 4*4*512])
    g_fcbias1 = initBias([4*4*512])

    g_convweight1 = initWeight([5,5,256,512])
    g_convweight2 = initWeight([5,5,128,256])
    g_convweight3 = initWeight([5,5,3,128])

    g_convbias1 = initBias([256])
    g_convbias2 = initBias([128])
    g_convbias3 = initBias([3])

    noise = tf.random_uniform([batchsize, 100], -1, 1)
    fc1 = tf.matmul(noise, g_fcweight1)
    relu1 = tf.nn.relu(fc1 + g_fcbias1)
    fattened = tf.reshape(relu1,[batchsize, 4, 4, 512])
    convt1 = tf.nn.conv2d_transpose(fattened, g_convweight1, [batchsize, 8, 8, 256], [1,2,2,1])
    relu2 = tf.nn.relu(convt1 + g_convbias1)
    convt2 = tf.nn.conv2d_transpose(relu2, g_convweight2, [batchsize, 16, 16, 128], [1,2,2,1])
    relu3 = tf.nn.relu(convt2 + g_convbias2)
    convt3 = tf.nn.conv2d_transpose(relu3, g_convweight3, [batchsize, 32, 32, 3], [1,2,2,1])
    relu4 = tf.nn.relu(convt3 + g_convbias3)
    return relu4;




def train():

    # prepare real data
    batch = unpickle("cifar-10-batches-py/data_batch_1")


    imagein = tf.placeholder("float",[batchsize,32,32,3])

    # discriminator called on real data
    d1 = discriminator(imagein)
    # keep track of which variables are for D and which are for G
    d_paramsnum = len(tf.trainable_variables())

    # discriminator called on generated images
    g = generator()
    d2 = discriminator(g)

    # parameters to train on for each respective update
    params = tf.trainable_variables()
    d_params = params[:d_paramsnum]
    g_params = params[d_paramsnum:]


    dloss = tf.reduce_mean(tf.nn.relu(d1) - d1 + tf.log(1.0 + tf.exp(-tf.abs(d1)))) + tf.reduce_mean(tf.nn.relu(d2) + tf.log(1.0 + tf.exp(-tf.abs(d2))))
    gloss = tf.reduce_mean(tf.nn.relu(d2) - d2 + tf.log(1.0 + tf.exp(-tf.abs(d2))))

    doptimizer = tf.train.AdamOptimizer(0.001).minimize(dloss, var_list=d_params)
    goptimizer = tf.train.AdamOptimizer(0.001).minimize(gloss, var_list=g_params)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(iterations):
            dloss = 0
            gloss = 0
            for k in xrange(updates):
                randomint = randint(0,10000 - batchsize - 1)
                trainingData = batch["data"][randomint:batchsize+randomint]
                rawlabel = batch["labels"][randomint:batchsize+randomint]
                trainingLabel = np.zeros((batchsize,10))
                trainingLabel[np.arange(batchsize),rawlabel] = 1
                trainingData = trainingData/255.0
                trainingData = np.reshape(trainingData,(batchsize,32,32,3))

                doptimizer.run(feed_dict= {imagein: trainingData})
                # dloss += dloss_delta
                print k

            goptimizer.run()
            print "i: " + str(k)
            saver.save(sess, os.getcwd()+"/training/train", global_step=i)
            # _, gloss_delta = goptimizer.run()
            # gloss += gloss_delta










train()
