# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:42:44 2023

@author: Snow
"""

# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score
import os
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from sklearn.metrics import f1_score


all_jaccard_scores_old = []
all_dice_coefficients_old = []
all_jaccard_scores_malignant = []
all_dice_coefficients_malignant = []


"""Define the image transformation pipeline"""

transform = transforms.Compose([
    transforms.ToTensor(),
])

"""Get the path of dataset"""

all_files = os.listdir("C:/Users/Snow/Desktop/ashwini/dataset/malignant/IMG_resized/")
all_files2 = os.listdir("C:/Users/Snow/Desktop/ashwini/dataset/benign/IMG_resized/")

image_path = "C:/Users/Snow/Desktop/ashwini/dataset/malignant/"
image_path2 = "C:/Users/Snow/Desktop/ashwini/dataset/benign/"


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
  
"""Instantiate the model"""

model = TransformerNetwork()

"""Load the model weights and set the model to evaluation mode"""

model_path = "C:/Users/Snow/Desktop/ashwini/transformer_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()


"""Threshold function to put pixel values as 0 or 1"""

def threshold_values(array):
    thresholded_array = np.where(array < 0.5, 0, 1)
    return thresholded_array


print("\n\n Processing Malignant Images !!!\n")

file_count = 1

start_time = time.time()

for file in all_files:
    
    file_parts = file.split('_')
    file_id = file_parts[1]
    original_image_path = image_path+"IMG_resized/"+str(file)
    input_image = Image.open(original_image_path)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    output = np.asarray(output)
    output = np.reshape(output, (output.shape[2], output.shape[3]))
    output = threshold_values(output)
    file_count+=1
    
    ground_truth_mask_path = image_path+"GT_resized/mali_"+str(file_id)+"_gt.png"
    ground_truth_mask = Image.open(ground_truth_mask_path)
    ground_truth_mask = np.asarray(ground_truth_mask)
    ground_truth_mask = threshold_values(ground_truth_mask)
    
    dice_coefficient = f1_score(output, ground_truth_mask, average='micro')
    dice_coefficient = round(dice_coefficient,2)
    all_dice_coefficients_old.append(dice_coefficient)
    
    jaccard_similarity = jaccard_score(output, ground_truth_mask, average='micro')
    jaccard_similarity = round(jaccard_similarity,2)
    all_jaccard_scores_old.append(jaccard_similarity)
    
    output_vector = np.reshape(output, (output.shape[0]*output.shape[1]))
    ground_truth_mask_vector = np.reshape(ground_truth_mask, (ground_truth_mask.shape[0]*ground_truth_mask.shape[1]))

    one_indices = []
    zero_indices = []
    
    for i in range(ground_truth_mask_vector.shape[0]):
        if(ground_truth_mask_vector[i] == 1):
            one_indices.append(i)
        else:
            zero_indices.append(i)
    
    background_prediction_vector = []
    background_gt_vector = []
    
    for i in zero_indices:
        background_gt_vector.append(ground_truth_mask_vector[i])
        background_prediction_vector.append(output_vector[i])
    
    dice_background = f1_score(background_gt_vector, background_prediction_vector, average='micro')
    dice_background = round(dice_background,2)
    all_dice_coefficients_malignant.append(dice_background)
    
    jaccard_background = jaccard_score(background_gt_vector, background_prediction_vector, average='micro')
    jaccard_background = round(jaccard_background,2)
    all_jaccard_scores_malignant.append(jaccard_background)
    
    
end_time = time.time()
total_time = end_time - start_time
total_time = round(total_time, 2)
print("total time: \t", total_time, " seconds \n")

print("Total malignant images read: \t", file_count, "\n")
all_dice_coefficients = np.asarray(all_dice_coefficients_old)
all_jaccard_scores = np.asarray(all_jaccard_scores_old)


"""Saving the Dice Coefficients & Jaccard Scores for malignant images and saving them on the disk and printing them"""

all_dice_coefficients_malignant = np.asarray(all_dice_coefficients_malignant)
all_jaccard_scores_malignant = np.asarray(all_jaccard_scores_malignant)
np.save("C:/Users/Snow/Desktop/ashwini/all_dice_coefficients_malignant_transformer.npy", all_dice_coefficients_malignant)
np.save("C:/Users/Snow/Desktop/ashwini/all_jaccard_scores_malignant_transformer.npy", all_jaccard_scores_malignant)

print("Mean (all_dice_coefficients_malignant): \t", np.round(np.mean(all_dice_coefficients_malignant),2), "\n")
print("SD (all_dice_coefficients_malignant): \t", np.round(np.std(all_dice_coefficients_malignant),2), "\n")
print("Mean (all_jaccard_scores_malignant): \t", np.round(np.mean(all_jaccard_scores_malignant),2), "\n")
print("SD (all_jaccard_scores_malignant): \t", np.round(np.std(all_jaccard_scores_malignant),2), "\n")


"""Getting the best segmentation results to plot on the basis of Dice Coefficients"""

index = np.argmax(all_dice_coefficients)
file = all_files[index]
file_parts = file.split('_')
file_id = file_parts[1]

full_image_path = image_path+"IMG_resized/"+file

best_malignant_image = Image.open(full_image_path)
best_malignant_image = np.asarray(best_malignant_image)

full_gt_path = image_path+"GT_resized/mali_"+str(file_id)+"_gt.png"
best_gt_image = Image.open(full_gt_path)
best_gt_image = np.asarray(best_gt_image)

best_malignant_image2 = Image.open(full_image_path)
best_malignant_image2 = transform(best_malignant_image2)

with torch.no_grad():
    best_prediction_image = model(best_malignant_image2)

best_prediction_image = np.asarray(best_prediction_image)
best_prediction_image = np.reshape(best_prediction_image, (best_prediction_image.shape[1], best_prediction_image.shape[2]))
best_prediction_image = threshold_values(best_prediction_image)

"""Creating the subplots to display and then saving the plot as PDF image"""

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(best_malignant_image)
axs[0].set_title('Original Malignant Image')

axs[1].imshow(best_gt_image, cmap='gray')
axs[1].set_title('Ground Truth')

axs[2].imshow(best_prediction_image)
axs[2].set_title('Prediction')

for ax in axs:
    ax.set_xlabel('X-axis')

fig.text(0.04, 0.5, 'Y-axis', va='center', rotation='vertical')

plt.tight_layout()
plt.savefig("C:/Users/Snow/Desktop/ashwini/Malignant_Graph.pdf")
plt.show()










"""==============================================================================================================="""












""" This is the experiments with the benign images"""

print("\n\nProcessing Benign Images !!! \n")

all_jaccard_scores_benign_old = []
all_dice_coefficients_benign_old = []
all_jaccard_scores_benign = []
all_dice_coefficients_benign = []

file_count = 1

start_time = time.time()

for file in all_files2:
        
    file_parts = file.split('_')
    file_id = file_parts[1]

    original_image_path = image_path2+"IMG_resized/"+str(file)
    input_image = Image.open(original_image_path)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    output = np.asarray(output)
    output = np.reshape(output, (output.shape[2], output.shape[3]))
    output = threshold_values(output)
    file_count+=1
    
    ground_truth_mask_path = image_path2+"GT_resized/beni_"+str(file_id)+"_gt.png"
    ground_truth_mask = Image.open(ground_truth_mask_path)
    ground_truth_mask = np.asarray(ground_truth_mask)
    ground_truth_mask = threshold_values(ground_truth_mask)
    
    dice_coefficient = f1_score(output, ground_truth_mask, average='micro')
    dice_coefficient = round(dice_coefficient,2)
    all_dice_coefficients_benign_old.append(dice_coefficient)
    
    jaccard_similarity = jaccard_score(output, ground_truth_mask, average='micro')
    all_jaccard_scores_benign_old.append(jaccard_similarity)
    
    output_vector = np.reshape(output, (output.shape[0]*output.shape[1]))
    ground_truth_mask_vector = np.reshape(ground_truth_mask, (ground_truth_mask.shape[0]*ground_truth_mask.shape[1]))  
    
    one_indices = []
    zero_indices = []
    
    for i in range(ground_truth_mask_vector.shape[0]):
        if(ground_truth_mask_vector[i] == 1):
            one_indices.append(i)
        else:
            zero_indices.append(i)
            
    background_prediction_vector = []
    background_gt_vector = []
    
    for i in zero_indices:
        background_gt_vector.append(ground_truth_mask_vector[i])
        background_prediction_vector.append(output_vector[i])
        
    dice_background = f1_score(background_gt_vector, background_prediction_vector, average='micro')
    dice_background = round(dice_background,2)
    all_dice_coefficients_benign.append(dice_background)
    
    jaccard_background = jaccard_score(background_gt_vector, background_prediction_vector, average='micro')
    jaccard_background = round(jaccard_background,2)
    all_jaccard_scores_benign.append(jaccard_background)
    

end_time = time.time()
total_time = end_time - start_time
total_time = round(total_time,2)
print("total time: \t", total_time, " seconds \n")

print("Total benign images read: \t", file_count, "\n")
all_dice_coefficients_benign = np.asarray(all_dice_coefficients_benign)
all_jaccard_scores_benign = np.asarray(all_jaccard_scores_benign)

"""Saving the Dice Coefficients onto the disk"""

np.save("C:/Users/Snow/Desktop/ashwini/all_dice_coefficient_benign_transformer.npy", all_dice_coefficients_benign)
np.save("C:/Users/Snow/Desktop/ashwini/all_jaccard_scores_benign_transformer.npy", all_jaccard_scores_benign)

all_dice_coefficients_benign = np.asarray(all_dice_coefficients_benign)
all_jaccard_scores_benign = np.asarray(all_jaccard_scores_benign)

np.save("C:/Users/Snow/Desktop/ashwini/all_dice_coefficients_benign_transformer.npy", all_dice_coefficients_benign)
np.save("C:/Users/Snow/Desktop/ashwini/all_jaccard_scores_benign_transformer.npy", all_jaccard_scores_benign)

print("Mean (all_dice_coefficients_benign): \t", np.round(np.mean(all_dice_coefficients_benign),2), "\n")
print("SD (all_dice_coefficients_benign): \t", np.round(np.std(all_dice_coefficients_benign),2), "\n")
print("Mean (all_jaccard_scores_benign): \t", np.round(np.mean(all_jaccard_scores_benign),2), "\n")
print("SD 0(all_jaccard_scores_benign): \t", np.round(np.std(all_jaccard_scores_benign),2), "\n")

"""Plotting the best image segmentation results"""
index = np.argmax(all_dice_coefficients_benign_old)
file = all_files2[index]
file_parts = file.split('_')
file_id = file_parts[1]

full_image_path = image_path2+"IMG_resized/"+file

best_benign_image = Image.open(full_image_path)
best_benign_image = np.asarray(best_benign_image)

full_gt_path = image_path2+"GT_resized/beni_"+str(file_id)+"_gt.png"
best_gt_image = Image.open(full_gt_path)
best_gt_image = np.asarray(best_gt_image)

best_benign_image2 = Image.open(full_image_path)
best_benign_image2 = transform(best_benign_image2)

with torch.no_grad():
    best_prediction_image = model(best_benign_image2)

best_prediction_image = np.asarray(best_prediction_image)
best_prediction_image = np.reshape(best_prediction_image, (best_prediction_image.shape[1], best_prediction_image.shape[2]))
best_prediction_image = threshold_values(best_prediction_image)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(best_benign_image)  
axs[0].set_title('Original Benign Image')

axs[1].imshow(best_gt_image)  
axs[1].set_title('Ground Truth')

axs[2].imshow(best_prediction_image)  
axs[2].set_title('Prediction')

for ax in axs:
    ax.set_xlabel('X-axis')

# Add a common y-axis label
fig.text(0.04, 0.5, 'Y-axis', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig("C:/Users/Snow/Desktop/ashwini/Benign_Graph.pdf")
plt.show()