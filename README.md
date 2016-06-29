# generative-adversial
Image generation via generative adversial networks on CIFAR-10, Imagenet, and Celeb faces written in Tensorflow.

This is code that goes along with [my post about generative adversial networks](http://kvfrans.com/generative-adversial-networks-explained/).

![](http://kvfrans.com/content/images/2016/06/cifar-early.png)

![](http://kvfrans.com/content/images/2016/06/cifar-late.png)

##How to use:

Download the datasets and place into the data folder.

Run `main.py` to start training, make sure to set `train = True` in the file.

Set `train = False` to visualize how adjusting the initial noise affects image generation.

Set `cifar = True` to train on CIFAR. This sets the network to train on 32x32 images instead of 64x64, and reads from the CIFAR binaries rather than from JPEGS.


Utils and network structure from [DCGAN-Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow).
