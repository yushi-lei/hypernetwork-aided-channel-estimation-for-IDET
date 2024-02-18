import numpy as np
import tensorflow as tf
from scipy.special import binom
from scipy.io import savemat
import datetime
import math
import hyperDNNs as hD
import hyperDNNs_FDD as hDF
import os

def Encoder(M,L,H,N0,batch_size):
    X_pilot_init = tf.Variable(tf.sqrt(1 / M) * (hD.standart_gaussian_noise_layer([M, 2 * L])), trainable=True , name='x')
    power_normal = tf.sqrt(tf.reduce_sum(tf.square(X_pilot_init[:, 0:L]) + tf.square(X_pilot_init[:, L:2 * L]), axis=0))
    X_pilot = X_pilot_init / (tf.concat([power_normal, power_normal], axis=0))
    X_tilde_complex = tf.complex(X_pilot[:, 0:L], X_pilot[:, L:2 * L])
    y = tf.matmul(H, X_tilde_complex)
    y_real = tf.concat([tf.real(y), tf.imag(y)], axis=1)
    noise = tf.sqrt(N0 / 2) * hD.standart_gaussian_noise_layer((batch_size, 2*L))
    y_noise = y_real + noise
    return y_noise,X_pilot

def hyper_aided_channel_estimation(M,Lp,batch_size,SNRdb,total_epoch,mini_batch,L,zeta):
    np.random.seed(0)
    K = 1
    P = 1.
    N_cov = 0.1 # circuit noise
    file_name = 'MYTRY(M=' + str(M) + ')(K=' + str(K) + ')(L=' + str(L) + ')(Lp=' + str(Lp) + ')(SNRdb=' + str(SNRdb) + ')(power_splitter_rate='+str(zeta)+')'
    print(file_name)
    input_number = M
    output_number = M * 2
    tf.reset_default_graph()
    H = tf.placeholder("complex64", [batch_size, input_number])
    y_true = tf.placeholder("float32", [batch_size, output_number])
    alpha_para = tf.placeholder("float32", [])
    N0_dnn = tf.placeholder("float32", [])


    y_noise,pilot = Encoder(M,L,H,N0_dnn,batch_size)
    y_pred = hD.hyperDNN(M, L, y_noise, zeta, N_cov)

    cross = tf.reduce_mean(tf.square(y_pred - y_true))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
    nmse = tf.reduce_mean(tf.square((y_pred - y_true)/y_true))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start_time = datetime.datetime.now()
    interval = (datetime.datetime.now() - start_time).seconds
    time_index = 1
    avg_cost = 0.
    avg_nmse = 0.
    cost_array = np.zeros((int(total_epoch / 10), 1), dtype=float)
    nmse_array = np.zeros((int(total_epoch / 10), 1), dtype=float)
    cost_index = 0
    with tf.Session() as sess:
        # Training
        sess.run(init)
        for epoch in range(total_epoch):

            if epoch < total_epoch / 5:
                learning_R = 0.001
            elif epoch < (3.0 * total_epoch / 4):
                learning_R = 0.0001
            else:
                learning_R = 0.00001
            SNR = 10 ** (SNRdb / 10.0)
            sigma2 = 1 / SNR
            # if epoch==0:           #FDD
            #    alpha_para_input=0.5
            # else:
            #    alpha_para_input=alpha_para_input*1.01
            # if alpha_para_input<10:
            #    alpha_para_input=10.0
            alpha_para_input = 1.0   #TDD
            for index_m in range(mini_batch):
                theta = 60.0 * np.random.rand(batch_size, K, Lp) - 30.0
                alpha = np.sqrt(0.5) * (np.random.standard_normal([batch_size, K, Lp]) + 1j * np.random.standard_normal(
                    [batch_size, K, Lp]))
                h = np.zeros([batch_size, M * 2], dtype=float)
                x_h = np.zeros([batch_size, M], dtype=complex)
                for p in range(Lp):
                    for m in range(M):
                        temp1 = np.exp(1j * 2 * np.pi * 0.5 * m * np.sin(theta[:, 0, p] / 180 * np.pi))
                        temp2 = 1.0 / np.sqrt(Lp) * alpha[:, 0, p] * temp1
                        x_h[:, m] = x_h[:, m] + temp2
                h[:, 0:M] = np.real(x_h)
                h[:, M:2 * M] = np.imag(x_h)
                N0_input = sigma2
                _, cs = sess.run([optimizer,cross],feed_dict={H: x_h, y_true: h, N0_dnn: N0_input, alpha_para: alpha_para_input,learning_rate: learning_R})
                nmse1=cs/np.mean(np.square(h))
                avg_cost += cs
                avg_nmse +=nmse1

            if (epoch + 1) % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_cost=", \
                      "{:.9f}".format(avg_cost / 10),"nmse:", '{:.9f}'.format(nmse1))

                cost_array[cost_index, 0] = avg_cost / 10
                nmse_array[cost_index, 0] = avg_nmse / 10
                cost_index += 1
                avg_cost = 0.
                avg_nmse = 0.
            if (epoch + 1) % 1000 == 0:
                train_val = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                a = sess.run(train_val)
                train_val2 = tf.get_collection(f.GraphKeys.TRAINABLE_VARIABLES)

                store_dic = {}
                store_dic['cost_array'] = cost_array
                store_dic['nmse_array'] = nmse_array
                for zz in range(len(train_val)):
                    dd = train_val[zz].name[0:-2]
                    dd = dd.replace("_normalization/", "_")
                    store_dic[dd] = np.array(a[zz])
                    if train_val2[-1].name == train_val[zz].name:
                        break
                savemat(file_name + ".mat", store_dic)

if __name__ == "__main__":
    hyper_aided_channel_estimation(M=64, Lp=2, batch_size=1024, SNRdb=10, total_epoch=10000, mini_batch=1, L=16, zeta=0.3)




