Ovarian Tumors Diagnosis and Characterization

Overview
This repository contains the code and resources for a project focused on diagnosing and characterizing ovarian tumors using CT scan images. The solution employs Ensemble Deep Learning techniques and Explainable AI to enhance the interpretability of the model's predictions.

Table of Contents

Introduction
Features
Installation
Usage
Model Architecture
Explainability
Results



Introduction
Ovarian tumors are a significant health concern, and timely and accurate diagnosis is crucial for effective treatment. This project introduces an innovative approach to ovarian tumor diagnosis, utilizing Ensemble Deep Learning for improved accuracy and Explainable AI to enhance transparency in the decision-making process.

Features
Ensemble Deep Learning: Combines multiple deep learning models for enhanced accuracy.
Explainable AI: Provides transparency into model predictions for better interpretability.
CT Scan Image Processing: Pre-processing techniques for improved model performance.
Diagnostic and Characterization Capabilities: Classifies ovarian tumors and provides detailed information on their characteristics.

Installation
1. Clone the Repository:
git clone https://github.com/Starlord33/ovarian-tumors-diagnosis.git
cd ovarian-tumors-diagnosis

2. Install Dependencies:
pip install -r requirements.txt

3. Download Data:
The dataset is available on a request basis.


Usage

1. Run the Main Application:
Refer to the Segmentation folder for the UNet and the Transformers.
Run the classification code.

Input CT scan images and review the diagnostic results provided by the system.

Model Architecture
The ensemble model comprises various deep learning architectures, including VGG-16, Inception V3, DenseNet-101, and ResNet-152. This combination enhances overall performance and robustness.


Explainability
Explainable AI techniques, such as SHAP & SmoothGrad, are employed to offer transparency into the model's decision-making process, ensuring that medical professionals can understand and trust the results produced.


Results
Detailed information on performance metrics, accuracy, and other relevant results achieved by the system on test datasets can be found here.