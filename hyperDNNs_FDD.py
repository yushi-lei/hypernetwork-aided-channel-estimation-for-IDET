import numpy as np
import tensorflow as tf
from scipy.special import binom
from tensorflow.contrib.layers import xavier_initializer
import math

def init_weights(shape,name): # 初始化权重，返回一个对应大小的浮点数变量
    return tf.get_variable(str(name)+'_w', shape, tf.float32, xavier_initializer())
def init_bias(shape,name): # 初始化偏置，返回一个对应大小的常数变量
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals,name=name+'_b')
def normal_full_layer(input_layer, size,name): # 根据输入层大小以及全连接层建点数初始化创建相应大小的权重与偏置，并且计算返回下一层
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size],name)
    b = init_bias([size],name)
    return tf.matmul(input_layer, W) + b
def standart_gaussian_noise_layer(shape): # 创建高斯分布的噪声，均值为0，方差为1
    noise = tf.random_normal(shape=shape)
    return noise

def frange(x, y, jump):
    while x <= y:
        yield x
        x += jump

def hyperDNN_FDD(M,L,y_noise,zeta,N_cov,B,alpha_para):
    #hyperDNN
    Efficiency=1
    P = Efficiency*(zeta)*(tf.square(y_noise[:, 0:L]) + tf.square(y_noise[:, L:(2 * L)]))
    l1 = (normal_full_layer(P, 128, 'l1'))
    n_l1 = tf.nn.relu(tf.layers.batch_normalization(l1, name='l1_normalization'))
    l2 = (normal_full_layer(n_l1, 512, 'l2'))
    n_l2 = tf.nn.relu(tf.layers.batch_normalization(l2, name='l2_normalization'))
    l3 = (normal_full_layer(n_l2, 1024, 'l3'))
    n_l3 = tf.nn.relu(tf.layers.batch_normalization(l3, name='l3_normalization'))
    wt = normal_full_layer(n_l3, (2*L + 128 + 512+1024), 'l4')
    #mainDNN
    y_noise_noise=np.sqrt(1-zeta)*y_noise  +tf.sqrt(N_cov/2)*standart_gaussian_noise_layer([1024,2*L])
    t1=(normal_full_layer(y_noise_noise*wt[:,0:2*L], 128, 't1'))
    n_t1 = tf.nn.relu(tf.layers.batch_normalization(t1, name='t1_normalization'))
    t2 = (normal_full_layer(n_t1*wt[:,2*L:(2*L+128)], 512, 't2'))
    n_t2 = tf.nn.relu(tf.layers.batch_normalization(t2, name='t2_normalization'))
    t3 = (normal_full_layer(n_t2*wt[:,(2*L+128):(2*L+128+512)], 1024, 't3'))
    n_t3 = tf.nn.relu(tf.layers.batch_normalization(t3, name='t3_normalization'))
    m1=(normal_full_layer(q, 1024, 'm1'))
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1, 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2, 256, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3, M * 2, 'm4')

    return h_est

def hyperDNN_FDD_old(M,L,y_noise,zeta,N_cov,B1,B2):
    #hyperDNN
    Efficiency=1
    P = Efficiency*(zeta)*(tf.square(y_noise[:, 0:L]) + tf.square(y_noise[:, L:(2 * L)]))
    l1 = (normal_full_layer(P, 128, 'l1'))
    n_l1 = tf.nn.relu(tf.layers.batch_normalization(l1, name='l1_normalization'))
    l2 = (normal_full_layer(n_l1, 512, 'l2'))
    n_l2 = tf.nn.relu(tf.layers.batch_normalization(l2, name='l2_normalization'))
    l3 = (normal_full_layer(n_l2, 1024, 'l3'))
    n_l3 = tf.nn.relu(tf.layers.batch_normalization(l3, name='l3_normalization'))
    q2 =tf.nn.tanh(normal_full_layer(n_l3,B2,'l4'))
    r1 =(normal_full_layer(q2,1024,'r1'))
    n_r1 = tf.nn.relu(tf.layers.batch_normalization(r1,name='r1_normalization'))
    r2 =(normal_full_layer(n_r1,512,'r2'))
    n_r2 = tf.nn.relu(tf.layers.batch_normalization(r2,name='r2_normalization'))
    r3 =(normal_full_layer(n_r2,512,'r3'))
    n_r3 = tf.nn.relu(tf.layers.batch_normalization(r3,name='r3_normalization'))
    wt = normal_full_layer(n_r3, (B1 + 1024 + 512 + 512), 'r4')
    #mainDNN
    y_noise_noise = np.sqrt(1-zeta)*y_noise+tf.sqrt(N_cov/2)*standart_gaussian_noise_layer([1024,2*L])
    t1=(normal_full_layer(y_noise_noise, 128, 't1'))
    n_t1 = tf.nn.relu(tf.layers.batch_normalization(t1, name='t1_normalization'))
    t2 = (normal_full_layer(n_t1, 512, 't2'))
    n_t2 = tf.nn.relu(tf.layers.batch_normalization(t2, name='t2_normalization'))
    t3 = (normal_full_layer(n_t2, 1024, 't3'))
    n_t3 = tf.nn.relu(tf.layers.batch_normalization(t3, name='t3_normalization'))
    q1 = tf.nn.tanh(normal_full_layer(n_t3, B1, 't4'))
    m1=(normal_full_layer(q1*wt[:,0:B1], 1024, 'm1'))
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1*wt[:,B1:(B1+1024)], 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2*wt[:,(B1+1024):(B1+1024+512)], 512, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3*wt[:,(B1+1024+512):(B1+1024+512+512)], M * 2, 'm4')
    return h_est

def mainDNN_FDD(M,L,y_noise,zeta,N_cov,B1,B2):
    #mainDNN
    y_noise_noise = np.sqrt(1 - zeta) * y_noise + tf.sqrt(N_cov/ 2)*standart_gaussian_noise_layer([1024, 2 * L])
    t1 = (normal_full_layer(y_noise_noise, 128, 't1'))
    n_t1 = tf.nn.relu(tf.layers.batch_normalization(t1, name='t1_normalization'))
    t2 = (normal_full_layer(n_t1, 512, 't2'))
    n_t2 = tf.nn.relu(tf.layers.batch_normalization(t2, name='t2_normalization'))
    t3 = (normal_full_layer(n_t2, 1024, 't3'))
    n_t3 = tf.nn.relu(tf.layers.batch_normalization(t3, name='t3_normalization'))
    q1 = tf.nn.tanh(normal_full_layer(n_t3, B1, 't4'))  # 为什么不用sign
    m1 = (normal_full_layer(q1, 1024, 'm1'))
    n_m1 = tf.nn.relu(tf.layers.batch_normalization(m1, name='m1_normalization'))
    m2 = (normal_full_layer(n_m1, 512, 'm2'))
    n_m2 = tf.nn.relu(tf.layers.batch_normalization(m2, name='m2_normalization'))
    m3 = (normal_full_layer(n_m2, 512, 'm3'))
    n_m3 = tf.nn.relu(tf.layers.batch_normalization(m3, name='m3_normalization'))
    h_est = normal_full_layer(n_m3, M * 2, 'm4')
    return h_est
