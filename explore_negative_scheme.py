import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def smooth(L, window=50):
    num = len(L)
    L1 = list(L[:window]) + list(L)
    out = [np.average(L1[i:i+window]) for i in range(num)]
    return np.array(out) if type(L) == type(np.array([0])) else out

## TopK similar, student fixed
# log_file_bas = 'crd_loss_log_s_001111.txt'
# log_file_K30 = 'crd_loss_log_s_235801.txt'
# log_file_K50 = 'crd_loss_log_s_235812.txt'
# log_file_K90 = 'crd_loss_log_s_235821.txt'
# crd_loss_bas = np.loadtxt(log_file_bas)
# crd_loss_K30 = np.loadtxt(log_file_K30)
# crd_loss_K50 = np.loadtxt(log_file_K50)
# crd_loss_K90 = np.loadtxt(log_file_K90)

## TopK similar, student not fixed
log_file_bas = 'crd_loss_log_s_170206.txt'
log_file_K30 = 'crd_loss_log_s_171853.txt'
log_file_K50 = 'crd_loss_log_s_170225.txt'
crd_loss_bas = np.loadtxt(log_file_bas)
crd_loss_K30 = np.loadtxt(log_file_K30)
crd_loss_K50 = np.loadtxt(log_file_K50)

## NCEK
# log_file_bas = 'crd_loss_log_s_231110.txt'
# log_file_NCEK100 = 'crd_loss_log_s_073230.txt'
# log_file_NCEK800 = 'crd_loss_log_s_010617.txt'
# crd_loss_bas = np.loadtxt(log_file_bas)
# crd_loss_NCEK100 = np.loadtxt(log_file_NCEK100)
# crd_loss_NCEK800 = np.loadtxt(log_file_NCEK800)

plt.subplot(2, 1, 1)
index = 3
# title = 'Neg pair loss'
title = 'Neg pair: exp(inner_product / t)'
plt.plot(smooth(crd_loss_bas[:, index]), label="base")
plt.plot(smooth(crd_loss_K50[:, index]), label="K50")
plt.plot(smooth(crd_loss_K30[:, index]), label="K30")
plt.legend()
plt.title(title)
plt.grid()

plt.subplot(2, 1, 2)
index = 2
# title = 'Pos pair loss'
title = 'Pos pair: exp(inner_product / t)'
plt.plot(smooth(crd_loss_bas[:, index]), label="base")
plt.plot(smooth(crd_loss_K50[:, index]), label="K50")
plt.plot(smooth(crd_loss_K30[:, index]), label="K30")
plt.legend()
plt.grid()
plt.title(title)
plt.show()