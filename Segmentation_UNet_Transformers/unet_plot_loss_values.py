# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:30:49 2023

@author: Snow
"""

import numpy as np
import matplotlib.pyplot as plt


loss_all_epochs = np.load("C:/Users/Snow/Desktop/ashwini2/unet_loss_all_epochs.npy")
x_axis = np.linspace(1,loss_all_epochs.shape[0], loss_all_epochs.shape[0])

plt.plot(x_axis, loss_all_epochs)
plt.title("Variation of Training Loss against Epochs", y=-0.3)
plt.ylabel("Training Loss")
plt.xlabel("Number of Epochs")

plt.savefig("C:/Users/Snow/Desktop/ashwini/unet_loss_all_epochs.pdf", format='pdf')
plt.show()


