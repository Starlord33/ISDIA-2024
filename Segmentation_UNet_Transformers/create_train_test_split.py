# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:10:58 2023

@author: Snow
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from PIL import Image



dataset_path = "C:/Users/Snow/Desktop/ashwini/dataset/"



all_features = []
all_labels = []



"""Reading BENIGN images and their labels"""


benign_file_count = 1
benign_files = os.listdir(dataset_path+"benign/IMG_resized/")

for benign_file in benign_files:
    
    print("Processing Benign Image: ", benign_file_count, "\n")
    
    benign_file_parts = benign_file.split('_')
    benign_file_id = benign_file_parts[1]
    
    image_path = dataset_path+"benign/IMG_resized/beni_"+str(benign_file_id)+"_img.png"
    img = Image.open(image_path)
    image_array = np.asarray(img)
    
    image_array = np.moveaxis(image_array, -1, 0)
    print("image_array.shape: \t", image_array.shape, "\n")
    
    all_features.append(image_array)
    
    label_path = dataset_path+"benign/GT_resized/beni_"+str(benign_file_id)+"_gt.png"
    lab = Image.open(label_path)
    label_array = np.asarray(lab)
    
    label_array = np.moveaxis(label_array, -1, 0)
    print("label_array.shape: \t", label_array.shape, "\n")
    all_labels.append(label_array)
    
    benign_file_count = benign_file_count + 1
    print("\n\n")
    



"""Reading MALIGNANT images and their labels"""


malignant_file_count = 1
malignant_files = os.listdir(dataset_path+"malignant/IMG/")

for malignant_file in malignant_files:
    
    print("Processing Malignant Image: ", malignant_file_count, "\n")
    
    malignant_file_parts = malignant_file.split('_')
    malignant_file_id = malignant_file_parts[1]
    
    image_path = dataset_path+"malignant/IMG_resized/mali_"+str(malignant_file_id)+"_img.png"
    img = Image.open(image_path)
    image_array = np.asarray(img)
    
    image_array = np.moveaxis(image_array, -1, 0)
    print("image_array.shape: \t", image_array.shape, "\n")
    
    all_features.append(image_array)
    
    label_path = dataset_path+"malignant/GT_resized/mali_"+str(malignant_file_id)+"_gt.png"
    lab = Image.open(label_path)
    label_array = np.asarray(lab)
    
    label_array = np.moveaxis(label_array, -1, 0)
    print("label_array.shape: \t", label_array.shape, "\n")
    all_labels.append(label_array)
    
    malignant_file_count = malignant_file_count + 1
    print("\n\n")
    

all_features = np.asarray(all_features)
all_labels = np.asarray(all_labels)

def threshold_values(array):
    # Set values less than 128 to 0 and values greater than or equal to 128 to 1
    thresholded_array = np.where(array < 128, 0, 1)
    return thresholded_array

all_labels = threshold_values(all_labels)

print("all_features.shape: \t", all_features.shape, "\n")
print("all_labels.shape: \t", all_labels.shape, "\n")


np.save("C:/Users/Snow/Desktop/ashwini/dataset/all_features_resized.npy", all_features)
np.save("C:/Users/Snow/Desktop/ashwini/dataset/all_labels_resized.npy", all_labels)


