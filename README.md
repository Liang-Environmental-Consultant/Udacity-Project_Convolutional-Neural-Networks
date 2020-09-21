# Data Science Nanodegree Project: Dog Breed Classification

##
Please follow the link for the Github: https://github.com/Liang-Environmental-Consultant/Udacity-Project_Convolutional-Neural-Networks

Please follow the link for the blog post: https://medium.com/@lilianli1986/a-study-of-parameters-influencing-dissolved-oxygen-concentration-in-a-stream-waterbody-in-florida-96f8038c170ahttps://medium.com/@lilianli1986/data-science-nanodegree-project-dog-breed-classification-ba82a6de36ea

## Project Overview

This is the Udacity Data Scientist Nanodegree project that classify dog and human images using Convolutional Neural Networks (CNN). In this project, a series of CNN models were designed and compared for image classification. With the optimized CNN model, the dog's breed can be estimated given an image of a dog. Also, the resembling dog breed can be estimated if an image of a human is given.        

## Software/Library Requirments
Python 3

sklearn

Numpy

TensorFlow

keras

glob

cv2

matplotlib

tqdm

PIL

## File Descriptions
dog_app.ipynb: Jupyter notebook containing implementation of CNN to classify breeds of dogs

dog_app.html: Jupyter notebook rendered in html format

test images (dog1.jpg, dog2.jpg, dog3.jpg, human1.jpg, human2.jpg, human3.jpg): These images are used to test the model


## Model Construction: CNN to Classify Dog Breeds (using Transfer Learning)

Transfer learning was applied using ResNet-50 bottleneck features to create a CNN that can identify dog breed from images.

### Model architecture:
Resnet50_model = Sequential()

Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))

Resnet50_model.add(Dense(133, activation='softmax'))

## Results
The CNN model can reach a test accuracy of 78.7%. Dog and human images can be appropriately predicted for breed. However, the model performance can be improved through the following ways: 

-Increase the number of training dateset;

-Hyper-parameter tunings: weight initializings, learning rates, batch_sizes, etc.;

-Using more complex architecture such as InceptionResNet.

