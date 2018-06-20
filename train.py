#running this file will train the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import os
import pickle
from datetime import datetime
from model import *
from ops import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_cifar10_data(filepath,filename,num_files,num_images):
    """ Loads the cifar-10 training data and labels and 
        val data and labels"""
    IMG_SIZE = 32
    CHANNELS = 3
    NUM_CLASSES = 10
    IMG_PER_FILE = 10000

    offset = (num_images%IMG_PER_FILE != 0)
    num_files_load = num_images//IMG_PER_FILE +offset

    images = np.zeros((num_images,IMG_SIZE,IMG_SIZE,CHANNELS),dtype=np.float32)
    labels = np.zeros(num_images,dtype = np.int32)
    begin = 0

    print("Loading Data")

    for i in range(num_files_load):
        
        if(filename != "test_batch"):
            datafile = filepath + filename + str(i+1)

        else:
            datafile = filepath + filename

        if(num_images<10000):
            img_slice = num_images
        else:
            img_slice = 10000
            num_images -= 10000

        with open(datafile,mode='rb') as file:
        
            raw_data = pickle.load(file,encoding='bytes')

            labels_data = np.array(raw_data[b'labels'])
        
            data = np.array(raw_data[b'data'],dtype=np.float32)/255.0

            images_data = np.reshape(data,(-1,IMG_SIZE,IMG_SIZE,CHANNELS))

            images = images_data[begin:begin+img_slice]
            labels = labels_data[begin:begin+img_slice]
            begin = img_slice
    #images = tf.convert_to_tensor(images,tf.float32)
    #labels = tf.convert_to_tensor(labels,tf.int32)

    print(images.shape,labels.shape)
    return images,labels

def run_model(data,args,conditional_input=None):
    X = tf.placeholder(tf.float32,shape=[None,args.height,args.width,args.channels])
    if args.condition:
        conditional_input = tf.placeholder(tf.int32,shape=[10,args.num_classes])
    #add conditional input
    
    loss,pred = PixelCNN(X,args,conditional_input)
    optimizer = tf.train.AdagradOptimizer(args.learning_rate)
    step = optimizer.minimize(loss)
    variables = [loss]
    if args.train:
        variables.append(step)

    train_indicies = np.arange(args.num_train).astype(np.int32)
    np.random.shuffle(train_indicies)    
    
    saver = tf.train.Saver(tf.trainable_variables())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.ckpt_file+'.meta'):
            saver = tf.train.import_meta_graph(args.ckpt_file+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_model))
            print("Model Restored")
        
        if args.train: 
            print("Model is training")
            for i in range(args.epoch):
                for j in range(args.batches-1):
                    print(j)
                    if args.data == "mnist":
                        x, y = data.train.next_batch(args.batch_size)
                        x = x.reshape([args.batch_size,args.height,args.width,args.channels])
                        #x = np.random.binomial(1,x).astype(np.float32)
                        x = (np.random.uniform(size=x.shape) < x).astype(np.float32)
                    else:
                        start = (j*args.batch_size)%args.num_train
                        idx = train_indicies[start:start+args.batch_size]
                        idx = idx[np.newaxis].T
                        x = np.take(data,idx,axis=0).reshape((args.batch_size,args.height,args.width,args.channels))

                    feed_dict = {X:x}
                    loss,_ = sess.run(variables,feed_dict = feed_dict)
                    
                    if(j%5 == 0):
                        print("iteration %d, Loss = %f" %(j,loss))
                    if (j+1)%60 == 0:
                        saver.save(sess, args.ckpt_file)
                    if (j+1)%300 == 0: 
                        saver.save(sess, args.ckpt_file)   
                        get_sample(sess,X,pred,args)
                print("epoch %d, Loss = %f" %(i,loss))
            get_sample(sess,X,pred,args)
        
        print("Validation")
        
        if args.data == "mnist":
            x,y = data.train.next_batch(args.batch_size)
            x = x.reshape([args.batch_size,args.height,args.width,args.channels])
                        #x = np.random.binomial(1,x).astype(np.float32)
            x = (np.random.uniform(size=x.shape) < x).astype(np.float32)
        else:
            start = (args.batches*args.batch_size)%args.num_train
            idx = train_indicies[start:start+args.batch_size]
            x = X[idx,:,:,:]
        feed_dict = {X:x}
        loss, = sess.run(variables[:1],feed_dict=feed_dict)
        print("Validation Loss = %f" %(loss))
            #generate samples at the end
        get_sample(sess,X,pred,args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--learning_rate',type=float,default=1e-3)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--train',type=lambda x: (str(x).lower() == 'true'),default=True)
    parser.add_argument('--condition', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='../datasets/mnist')
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--samples_path', type=str, default='samples')
    parser.add_argument('--summary_path', type=str, default='logs')
    args = parser.parse_args()
    
    if args.data == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)
        data = input_data.read_data_sets(args.data_path)
        args.num_classes = 10
        args.num_train = 60000
        args.height = 28
        args.width = 28
        args.channels = 1
        args.batches = data.train.num_examples // args.batch_size
    else:
        datapath = "../datasets/cifar-10-batches-py/"
        num_files = 5
        args.num_train = 10000

        filename = "data_batch_"
        data,labels = load_cifar10_data(datapath,filename,num_files,args.num_train)
        mean = np.mean(data,0)
        data -= mean
        args.num_classes = 10
        args.height = 32
        args.width = 32
        args.channels = 3
        args.batches = args.num_train // args.batch_size 
    
    #creating required files

    args.ckpt_model = os.path.join(args.ckpt_path,"%s_%d_%d"%(args.data,args.num_layers,args.num_filters))
    if not os.path.exists(args.ckpt_model):
        os.makedirs(args.ckpt_model)
    args.ckpt_file = os.path.join(args.ckpt_model,"model.ckpt")
    args.samples_path = os.path.join(args.samples_path, "%d_%d_%d"%(args.epoch, args.num_layers, args.num_filters))
    if not os.path.exists(args.samples_path):
        os.makedirs(args.samples_path)

    if tf.gfile.Exists(args.summary_path):
        tf.gfile.DeleteRecursively(args.summary_path)
    tf.gfile.MakeDirs(args.summary_path)

    if args.condition:
        run_model(data,args,conditional_input=None) #add conditional input here
    else:
        run_model(data,args)