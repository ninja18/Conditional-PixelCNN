import tensorflow as tf
import numpy as np
from ops import *

#This file contains the pixelcnn model
def init(params,num_layers):
    for layer in range(num_layers):
        params['wf' + str(layer)] = None
        params['wg' + str(layer)] = None
        params['bf' + str(layer)] = None
        params['bg' + str(layer)] = None
        params['w'+'h' + str(layer)] = None
        params['w'+'v' + str(layer)] = None
        params['b'+'h' + str(layer)] = None
        params['b'+'v' + str(layer)] = None
    return params
def PixelCNN(X,args,conditional_input):
    v_in,h_in = X,X
    params = {}
    params = init(params,args.num_layers)
    for i in range(args.num_layers):
        mask = 'b'
        filter_size = 3
        residual_con = True
        channel = args.num_filters
        if(i == 0):
            mask = 'a'
            filter_size = 7
            residual_con = False
            channel = 1 if args.data == 'mnist' else 3
        with tf.variable_scope("v_stack_in"+str(i)):
            v_stack_in = Gate(v_in,[filter_size,filter_size,channel,args.num_filters],i,mask,'v',args.condition,conditional_input).gated_conv()
            v_in = v_stack_in

        with tf.variable_scope("v_stack_out"+str(i)):
            v_stack_out = Gate(v_in,[1,1,args.num_filters,args.num_filters],i,mask,'v',mode='standard').conv()
            
        with tf.variable_scope("h_stack_in"+str(i)):
            h_stack_in = Gate(h_in,[filter_size,filter_size,channel,args.num_filters],i,mask,'h',args.condition,conditional_input).gated_conv()
            h_stack_in += v_stack_out
            
        with tf.variable_scope("h_stack_out"+str(i)):
            h_stack_out = Gate(h_stack_in,[1,1,args.num_filters,args.num_filters],i,mask,'h',mode='standard').conv()
            
        if(residual_con):
            h_stack_out += h_in
        h_in = h_stack_out
        
    with tf.variable_scope("conv1"):
        conv1 = Gate(h_in,[1,1,args.num_filters,args.num_filters],1,'b','l',mode='standard').conv()
        aconv1 = tf.nn.relu(conv1)

    if args.data == 'mnist':
        with tf.variable_scope("conv2"):
            conv2 = Gate(aconv1,[1,1,args.num_filters,1],2,'b','l',mode='standard').conv()
            
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv2,labels=X))
        pred = tf.nn.sigmoid(conv2)
      
    else:
        with tf.variable_scope("conv2"):
            conv2 = Gate(conv1,[1,1,args.num_filters,3*256],2,'b','l',mode='standard').conv()
            conv2 = tf.reshape(conv2,(-1,256))
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=conv2,labels=tf.cast(tf.reshape(X,[-1]),dtype=tf.int32)))
        pred = tf.reshape(tf.argmax(tf.nn.softmax(conv2),axis = 1),tf.shape(X))
        
    return loss,pred