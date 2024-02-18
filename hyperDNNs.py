import numpy as np
import tensorflow as tf
from scipy.special import binom
from tensorflow.contrib.layers import xavier_initializer
import math

def init_weights(shape,name):
    return tf.get_variable(str(name)+'_w', shape, tf.float32, xavier_initializer())
def init_bias(shape,name):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals,name=name+'_b')
def normal_full_layer(input_layer, size,name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size],name)
    b = init_bias([size],name)
    return tf.matmul(input_layer, W) + b
def standart_gaussian_noise_layer(shape):
    noise = tf.random_normal(shape=shape)
    return noise

def frange(x, y, jump):
    while x <= y:
        yield x
        x += jump

def hyperDNN(M,L,y_noise,zeta,N_cov):
    #hyper
    Efficiency = 1
    P = Efficiency * (zeta) * (tf.square(y_noise[:, 0:L]) + tf.square(y_noise[:, L:(2 * L)]))
    l1 = (normal_full_layer(P, 128, 'l1'))
    n_l1 = tf.nn.relu(tf.layers.batch_normalization(l1, name='l1_normalization'))
    #l2 = (normal_full_layer(n_l1, 256, 'l2'))
    #n_l2 = tf.nn.relu(tf.layers.batch_normalization(l2, name='l2_normalization'))
    l3 = (normal_full_layer(n_l1, 512, 'l3'))
    n_l3 = tf.nn.relu(tf.layers.batch_normalization(l3, name='l3_normalization'))
    l4 = (normal_full_layer(n_l3, 1024, 'l4'))
    n_l4 = tf.nn.relu(tf.layers.batch_normalization(l4, name='l4_normalization'))
    wt = normal_full_layer(n_l4, (2 * L + 1024 + 512 + 512), 'l5')
    #main DNN
    y_noise_noise=np.sqrt(1 - zeta) * y_noise + tf.sqrt(N_cov / 2) * standart_gaussian_noise_layer([1024,2 * L])
    m1 = (normal_full_layer(y_noise_noise * wt[:, 0:2 * L], 1024, 'm1'))
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1 * wt[:, 2*L:(2*L+1024)], 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2 * wt[:, (2*L+1024):(2*L+1024+512)], 512, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3 * wt[:, (2*L+1024+512):(2*L+1024+512+512)], M * 2, 'm4')
    return h_est

def mainDNN(M,L,y_noise,zeta,N_cov):
    #main DNN
    y_noise_noise = np.sqrt(1 - zeta) * y_noise + tf.sqrt(N_cov/ 2)*standart_gaussian_noise_layer([1024, 2 * L])
    m1 = (normal_full_layer(y_noise_noise, 1024, 'm1'))
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1, 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2, 512, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3, M * 2, 'm4')
    return h_est

def hyperDNN_timeswitching(M,L,y_noise,zeta,N_cov): #time switching mode
    #hyperDNN
    L2 = int(L * eta)
    L1 = L - L2
    Efficiency = 1
    P = Efficiency* zeta * (tf.square(y_noise[:, L1:L]) + tf.square(y_noise[:, (L+L1):(2 * L)]))
    l1 = (normal_full_layer(P, 128, 'l1'))
    n_l1 = tf.nn.relu(tf.layers.batch_normalization(l1, name='l1_normalization'))
    #l2 = (normal_full_layer(n_l1, 256, 'l2'))
    #n_l2 = tf.nn.relu(tf.layers.batch_normalization(l2, name='l2_normalization'))
    l3 = (normal_full_layer(n_l1, 512, 'l3'))
    n_l3 = tf.nn.relu(tf.layers.batch_normalization(l3, name='l3_normalization'))
    l4 = (normal_full_layer(n_l3, 1024, 'l4'))
    n_l4 = tf.nn.relu(tf.layers.batch_normalization(l4, name='l4_normalization'))
    wt = normal_full_layer(n_l4, (2 * L1 + 1024 + 512 + 512), 'l5')
    #mainDNN
    y_noise_noise = np.sqrt(1-zeta)*y_noise+tf.sqrt(N_cov/2)*standart_gaussian_noise_layer([1024,2*L])
    m1_1 = (normal_full_layer(y_noise[:,0:L1]*wt[:,0:L1], 512, 'm1_1'))
    m1_2 = (normal_full_layer(y_noise[:, L:(L1+L)] * wt[:, L1:2*L1], 512, 'm1_2'))
    m1=tf.concat([m1_1,m1_2],axis=1)
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1*wt[:,2*L1:(2*L1+1024)], 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2*wt[:,(2*L1+1024):(2*L1+1024+512)], 512, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3*wt[:,(2*L1+1024+512):(2*L1+1024+512+512)], M * 2, 'm4')
    return h_est

def mainDNN_timeswitching(M,L,y_noise,zeta,N0):
    #mainDNN
    L2 = int(L * zeta)
    L1 = L - L2
    m1_1 = (normal_full_layer(y_noise[:, 0:L1], 512, 'm1_1'))
    m1_2 = (normal_full_layer(y_noise[:, L:(L1 + L)], 512, 'm1_2'))
    m1 = tf.concat([m1_1, m1_2], axis=1)
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1, 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2, 512, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3, M * 2, 'm4')
    return h_est
