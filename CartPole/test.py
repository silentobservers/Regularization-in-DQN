# Author: Chengjia Lei

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure as figure

########################### Load Data ################################
x_axis = np.load('./base/50000/x_axis.npy')
base_y_mean = np.load('./base/50000/y_mean.npy')
base_y_std = np.load('./base/50000/y_std.npy')
L1_y_mean = np.load('./L1/50000/y_mean.npy')
L1_y_std = np.load('./L1/50000/y_std.npy')
L2_y_mean = np.load('./L2/50000/y_mean.npy')
L2_y_std = np.load('./L2/50000/y_std.npy')
dropout_y_mean = np.load('./dropout/50000/b/y_mean.npy')
dropout_y_std = np.load('./dropout/50000/b/y_std.npy')
dropout_y_mean_o = np.load('./dropout_o/50000/y_mean.npy')
dropout_y_std_o = np.load('./dropout_o/50000/y_std.npy')
dropout_y_mean_n = np.load('./new_dropout/50000/y_mean.npy')
dropout_y_std_n = np.load('./new_dropout/50000/y_std.npy')

############################ Plot #################################
plt.plot(x_axis, base_y_mean, lw=2, linestyle='-', label='Base', color='tomato')
plt.fill_between(x_axis, base_y_mean + base_y_std/2, base_y_mean - base_y_std/2, facecolor='tomato', alpha=0.2)
plt.plot(x_axis, L1_y_mean, lw=2, linestyle='--', label='L1', color='gold')
plt.fill_between(x_axis, L1_y_mean + L1_y_std/2, L1_y_mean - L1_y_std/2, facecolor='gold', alpha=0.2)
plt.plot(x_axis, L2_y_mean, lw=2, linestyle=':', label='L2', color='cyan')
plt.fill_between(x_axis, L2_y_mean + L2_y_std/2, L2_y_mean - L2_y_std/2, facecolor='cyan', alpha=0.2)
plt.plot(x_axis, dropout_y_mean_o, lw=2, linestyle='-.', label='Dropout', color='blue')
plt.fill_between(x_axis, dropout_y_mean_o + dropout_y_std_o/2, dropout_y_mean_o - dropout_y_std_o/2, facecolor='blue', alpha=0.2)
plt.plot(x_axis, dropout_y_mean, lw=2, linestyle='-.', label='Improved dropout', color='green')
plt.fill_between(x_axis, dropout_y_mean + dropout_y_std/2, dropout_y_mean - dropout_y_std/2, facecolor='green', alpha=0.2)
plt.plot(x_axis, dropout_y_mean_n, lw=2, linestyle='-.', label='New dropout', color='purple')
plt.fill_between(x_axis, dropout_y_mean_n + dropout_y_std_n/2, dropout_y_mean_n - dropout_y_std_n/2, facecolor='purple', alpha=0.2)

plt.ylim(0,1100)
plt.xlabel('Pole Length')
plt.ylabel('Rewards')

plt.title('CartPole (50000 steps)')
plt.legend()
plt.grid(linestyle='--')
plt.show()
