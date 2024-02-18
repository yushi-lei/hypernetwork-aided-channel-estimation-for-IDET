import numpy as np
from scipy.special import binom
import math
from scipy.linalg import pinv

def MMSE_channel_estimation(M,Lp,batch_size,SNRdb,mini_batch,L):
    np.random.seed(0)
    K = 1
    file_name = 'ZF_CE(M=' + str(M) + ')(K=' + str(K) + ')(L=' + str(L) + ')(Lp=' + str(Lp) + ')(SNRdb=' + str(SNRdb) + ')'
    print(file_name)
    SNR = 10 ** (SNRdb / 10.0)
    sigma2 = 1 / SNR
    N0_dnn=sigma2
    avg_cost1=0.
    avg_nmse1=0.
    for index_m in range(mini_batch):
        theta = 60.0 * np.random.rand(batch_size, K, Lp) - 30.0
        alpha = np.sqrt(0.5) * (np.random.standard_normal([batch_size, K, Lp]) + 1j * np.random.standard_normal(
            [batch_size, K, Lp]))
        x_h = np.zeros([batch_size, M], dtype=complex)
        for p in range(Lp):
            for m in range(M):
                temp1 = np.exp(1j * 2 * np.pi * 0.5 * m * np.sin(theta[:, 0, p] / 180 * np.pi))
                temp2 = 1.0 / np.sqrt(Lp) * alpha[:, 0, p] * temp1
                x_h[:, m] = x_h[:, m] + temp2
        Rhh=np.corrcoef(x_h.T)
        X_pilot_init = np.sqrt(1 / M) * np.random.standard_normal([M, 2 * L])
        power_normal = np.sqrt(np.sum(np.square(X_pilot_init[:, 0:L]) + np.square(X_pilot_init[:, L:2 * L]), axis=0))
        X_pilot = X_pilot_init / (
            np.concatenate([power_normal, power_normal], axis=0))

        X_tilde_complex = X_pilot[:, 0:L] + X_pilot[:, L:2 * L] * 1j
        XT=X_tilde_complex.T
        HHX=np.matmul(XT.conj().T,XT)
        HHX1=pinv(pinv(HHX)*sigma2*np.eye(M)+Rhh)
        y = np.matmul(XT, x_h.T)
        noise = np.sqrt(N0_dnn / 2) * np.random.standard_normal([L,batch_size]) + 1j * np.sqrt(N0_dnn / 2) * np.random.standard_normal([L,batch_size])
        y_noise = y + noise
        h_ls=np.matmul(pinv(XT),y_noise)
        HHX2=np.matmul(HHX1,Rhh)
        h_mmse=np.matmul(HHX2,h_ls)
        h_ls_r = np.concatenate([np.real(h_ls), np.imag(h_ls)], axis=0)
        h_mmse_r = np.concatenate([np.real(h_mmse), np.imag(h_mmse)], axis=0)
        x_h_r = np.concatenate([np.real(x_h), np.imag(x_h)], axis=1)
        cost1 = np.mean(np.square(h_mmse_r - x_h_r.T))
        nmse1 = cost1 / np.mean(np.square(x_h_r.T))
        avg_cost1+=cost1
        avg_nmse1+=nmse1
        print(avg_cost1)
        print(avg_nmse1)



