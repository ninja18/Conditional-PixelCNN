#This file contains the operations required for pixelcnn

import tensorflow as tf
import numpy as np
from datetime import datetime
import scipy.misc
import os

class  Gate(object):
    """"gate"""
    def __init__(self,X,shape,layer,mask,stack_type,condition=False,conditional_input=None,mode=None):
        
        self.X = X
        self.shape = shape
        self.layer = layer
        self.mask = mask
        self.stack_type = stack_type
        self.condition = condition
        self.conditional_input = conditional_input
        self.mode = mode
        
    def gated_conv(self):
        """performs gated convolution for the given input and if conditioned it performs conditional gated convolution
        """
        wf = tf.get_variable("wf",self.shape,tf.float32)
        wg = tf.get_variable("wg",self.shape,tf.float32)
        wf = masked_weights(wf,self.shape,self.mask,self.stack_type)
        wg = masked_weights(wg,self.shape,self.mask,self.stack_type)
        
        bf = get_bias("bf",self.shape[3],self.condition,self.conditional_input)
        bg = get_bias("bg",self.shape[3],self.condition,self.conditional_input)
        
        term1 = tf.nn.conv2d(self.X,wf,strides=[1,1,1,1],padding='SAME')
        term2 = tf.nn.conv2d(self.X,wg,strides=[1,1,1,1],padding='SAME')
        
        aterm1 = tf.tanh(term1+bf)
        aterm2 = tf.sigmoid(term2+bg)
        
        return tf.multiply(aterm1,aterm2)

    def conv(self):
        
        w = tf.get_variable("w",self.shape,tf.float32)
        w = masked_weights(w,self.shape,self.mask,self.stack_type,mode=self.mode)
        
        b = tf.get_variable("b",self.shape[3],tf.float32,tf.zeros_initializer)
        out = tf.nn.conv2d(self.X,w,strides=[1,1,1,1],padding='SAME')
        
        return tf.add(out,b)

def masked_weights(weight,shape,mask,stack_type,mode=None):
    if mask:
        midx = shape[1]//2
        midy = shape[0]//2
        masker = np.ones(shape,dtype=np.float32)
        if mode == "standard":
            masker[midy,midx+1:,:,:] = 0.0
            masker[midy+1:,:,:] = 0.0
            if mask == 'a':
                masker[midy,midx,:,:] = 0.0

        else:
            if mask == 'a':
                masker[midy,midx,:,:] = 0.0

            if stack_type == 'h':
                masker[midy,midx+1:,:,:] = 0.0
                masker[midy+1:,:,:] = 0.0
            else:
                if mask == 'a':
                    masker[midy:,:,:,:] = 0.0
                else:
                    masker[midy+1:,:,:,:] = 0.0
        masked_weight = weight * masker
    return masked_weight

def get_bias(b,shape,conditional,conditional_input):
    b = tf.get_variable(b,shape,tf.float32,tf.zeros_initializer)
    return b

def get_sample(sess,X, pred, args,conditional_input=None):
    print("Generating Sample Images...")
    n_row, n_col = 3,3
    samples = np.zeros((n_row*n_col, args.height, args.width, args.channels), dtype=np.float32)
    labels = tf.one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), args.num_classes)
    for i in range(args.height):
        for j in range(args.width):
            for k in range(args.channels):
                feed_dict = {X:samples}
                if args.condition is True:
                    data_dict[conditional_input] = labels
                next_sample = sess.run(pred, feed_dict=feed_dict)
                if args.data == "mnist":
                    next_sample = np.random.binomial(1,next_sample).astype(np.float32)
                samples[:, i, j, k] = next_sample[:, i, j, k]
    print("samples generated...")
    save_images(samples, n_row, n_col, args)

def save_images(samples, n_row, n_col, args):
    images = samples 
    if args.data == "mnist":
        images = images.reshape((n_row, n_col, args.height, args.width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((args.height * n_row, args.width * n_col))
    else:
        images = images.reshape((n_row, n_col, args.height, args.width, args.channels))
        images = images.transpose(1, 2, 0, 3, 4)
        images = images.reshape((args.height * n_row, args.width * n_col, args.channels))

    filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+".jpg"
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(args.samples_path, filename))
    print("samples saved...")