"""
@AmineHorseman
Sep, 1st, 2016
"""
import tensorflow as tf 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.merge_ops import merge_outputs, merge
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression 
from tflearn.optimizers import Momentum, Adam

from parameters import NETWORK, HYPERPARAMS

def build_model(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):

    images_input = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input1')
    
    images_network = conv_2d(images_input, 16, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 16, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides=2)  #24*24*16
    
    
    images_network = conv_2d(images_network, 32, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 32, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides=2)    #12*12*32
    
    images_network=tf.pad(images_network,[[0,0],[18,18],[18,18],[0,0]],'CONSTANT')
    images_network = merge([images_network, images_input], 'concat', axis=3)              #48*48*33
    
    images_network = conv_2d(images_network, 64, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 64, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 64, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides=2)       #24*24*64
    
    
    images_network = conv_2d(images_network, 128, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 128, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 128, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides=2)      #12*12*128
#     
    images_network = conv_2d(images_network, 128, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 128, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = conv_2d(images_network, 128, 3, activation='relu')
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 2, strides=2)     #6*6*128
     
    images_network = fully_connected(images_network, 1024, activation='relu')
    images_network = dropout(images_network,keep_prob=keep_prob)
    if NETWORK.use_batchnorm_after_fully_connected_layers:
        images_network = batch_normalization(images_network)
    images_network = fully_connected(images_network, 1024, activation='relu')
    images_network = dropout(images_network, keep_prob=keep_prob)
    if NETWORK.use_batchnorm_after_fully_connected_layers:
        images_network = batch_normalization(images_network)

    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
        if NETWORK.use_hog_sliding_window_and_landmarks:
            landmarks_network = input_data(shape=[None, 2728], name='input2')
        elif NETWORK.use_hog_and_landmarks:
            landmarks_network = input_data(shape=[None, 208], name='input2')
        else:
            landmarks_network = input_data(shape=[None, 68, 2], name='input2')
        landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation)
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)
        landmarks_network = fully_connected(landmarks_network, 40, activation=NETWORK.activation)
        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)
        images_network = fully_connected(images_network, 40, activation=NETWORK.activation)
        network = merge([images_network, landmarks_network], 'concat', axis=1)
    else:
        network = images_network
    network = fully_connected(network, NETWORK.output_size, activation='softmax')

    if optimizer == 'momentum':
        # FIXME base_lr * (1 - iter/max_iter)^0.5, base_lr = 0.01
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
                    lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print("Unknown optimizer: {}".format(optimizer))
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network