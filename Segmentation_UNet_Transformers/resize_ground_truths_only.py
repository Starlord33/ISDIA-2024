# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:15:34 2023

@author: Snow
"""

from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size=(256, 256)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    for file_name in files:
        # Construct the full path for each file
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            # Open the image and convert it to grayscale
            with Image.open(input_path).convert('L') as img:
                # Resize the single-channel (grayscale) image
                resized_img = img.resize(new_size)
                # Save the resized image to the output folder
                resized_img.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    # Specify your input and output folders
    input_folder = "C:/Users/Snow/Desktop/ashwini/dataset/malignant/GT/"
    output_folder = "C:/Users/Snow/Desktop/ashwini/dataset/malignant/GT_resized/"

    # Resize images (as single-channel) and save to the output folder
    resize_images(input_folder, output_folder)
