# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:39:12 2023

@author: Snow
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:52:51 2023

@author: Snow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
import time


torch.cuda.empty_cache() 


x_train = np.load("C:/Users/Snow/Desktop/ashwini/dataset/all_features_resized.npy")
y_train = np.load("C:/Users/Snow/Desktop/ashwini/dataset/all_labels_resized.npy")


x_train, y_train = shuffle(x_train, y_train, random_state=0)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

x_train = x_train.float()
y_train = y_train.float()
x_train = x_train/255.0


batch_size = 4

# Create a TensorDataset
dataset = TensorDataset(x_train, y_train)

# Create a DataLoader for batching
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Total number of batches: \t", len(train_loader), "\n")

device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim
import time




import torch.nn as nn

class TransformerNetwork(nn.Module):
    def __init__(self):
        super(TransformerNetwork, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        )

        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        # Apply sigmoid activation to squash the output values to the range [0, 1]
        x2 = self.sigmoid(x2)
        return x2



# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the UNet model
model = TransformerNetwork().to(device)

# Print the model summary
torchsummary.summary(model, (3, 256, 256), device=str(device))

criterion = nn.BCELoss()  # Binary Cross Entropy with Logits loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.01)

start_time = time.time()

num_epochs = 1

loss_all_epochs = []

for epoch in range(num_epochs):
    
    model.train()  
    epoch_loss = 0.0

    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        x,y,z = targets.size()
        targets = targets.reshape(x,1,y,z)

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()

    epoch_loss = round(epoch_loss,3)
    loss_all_epochs.append(epoch_loss)
    print("Epoch: ", epoch+1, "\t Loss: ", epoch_loss, "\n")

end_time = time.time()
total_training_time = end_time - start_time
total_training_time = round(total_training_time,2)
print("Total training time: ", total_training_time, " seconds.")


model.eval() 
input_tensor = x_train[5]
model.to('cpu')

with torch.no_grad():
    output_tensor = model(input_tensor)
    
output_prediction = np.asarray(output_tensor)

print("output_prediction before quantization: ", output_prediction, "\n")


def threshold_values(array):
    # Set values less than 128 to 0 and values greater than or equal to 128 to 1
    thresholded_array = np.where(array < 0.4, 0, 1)
    return thresholded_array

output_prediction = threshold_values(output_prediction)
print("output_prediction after quantization: ", output_prediction, "\n")


print("np.unique(output_prediction): \t", np.unique(output_prediction), "\n")

import matplotlib.pyplot as plt

plt.imshow(output_prediction.T, cmap='gray')
plt.title("Sample Input/Output")


output_prediction_resized = np.reshape(output_prediction, (output_prediction.shape[1], output_prediction.shape[2]))

from sklearn.metrics import jaccard_score

result = jaccard_score(y_train[5], output_prediction_resized, average='micro')
print("Jaccard Score: \t", result, "\n")

torch.save(model.state_dict(), "C:/Users/Snow/Desktop/ashwini2/transformer_model.pth")

loss_all_epochs = np.asarray(loss_all_epochs)
np.save("C:/Users/Snow/Desktop/ashwini2/transformer_loss_all_epochs.npy", loss_all_epochs)
