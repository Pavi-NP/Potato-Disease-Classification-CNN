#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:58:51 2024

@author: paviprathiraja

Potato Disease Classification

Download dataset into TF data pipeline and do data cleaning and ready the dataset for modelling

First part of the code provides a comprehensive pipeline for preparing a dataset for modeling, 
including data loading, exploration, visualization, partitioning, 
and optimization for efficient training.
#######
Second part of the code snippet demonstrates the process of building, training, and evaluating 
a convolutional neural network model for classifying images of potato diseases, 
including data preprocessing, augmentation, model architecture definition, compilation, 
training, and evaluation.
"""

#-------------------IMPORT LIBRARIES -----------------

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# use tensorflow dataset to download all the images to tf.data.dataset pipeline
#------------------------------
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS =3 #RGB channels
EPOCHS =50

# ----------------------Download Dataset -------------------------- 
# The dataset is downloaded using TensorFlow's image_dataset_from_directory function.

dataset= tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle="True",
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
    )
#---------------Explore the data set ------------------------------------------
# Explore the Dataset: Various aspects of the dataset are explored, such as the class names, 
# the length of the dataset, and the shape of the image batches.

class_names = dataset.class_names
print(class_names)   # 0, 1,2

length=len(dataset)
print(length)

#-----------------------Visualize Images--------------------------------
for image_batch, label_batch in dataset.take(1): # image_batch is one batch it includes 32 images
    print(image_batch.shape)
    print(label_batch.numpy())

# (32, 256, 256, 3)  3 for RGB
# [1 0 0 1 1 0 1 1 0 0 1 0 0 0 1 0 1 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1] 


    print(image_batch[0])
    print(image_batch[0].numpy())
    print(image_batch[0].shape)
    
# # VISUALIZING the images
# for image_batch, label_batch in dataset.take(1):
#     plt.imshow(image_batch[0].numpy().astype("uint8"))
#     plt.title(class_names[label_batch[0]])
#     plt.axis("off")
#     print(image_batch[0].shape)
    
#----------------------------------------------------------------------------  
plt.Figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):

# VISUALIZING the images
    for i in range(12):
        ax = plt.subplot(4,3,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    print(image_batch[0].shape)

print(len(dataset))
# # length of dataset is 68. but actual dataset size is 68x32

# """
# 80% ==> training
# 20% ==>  10% validaton,  10%  test

# -------------- Partition the Dataset --------------------
# The dataset is split into training, validation, and test sets. 80% of the data is allocated 
# for training, and the remaining 20% is divided equally between validation and test sets.
# # train dataset
train_size = 0.8 
len(dataset)* train_size  # = 54.400000
print(len(dataset) * train_size)


train_ds = dataset.take(54) # this is like first 54 batches i.e. arr[:54]
print(len(train_ds))

test_ds = dataset.skip(54) # skipping first 54 and taking next i.e. arr[54:] 54 onwards
print(len(test_ds))

# validatin dataser
val_size = 0.1 # validation data size is 10% = 6.80000
val_ds = test_ds.take(6)
print(len(val_ds))

#actual test data set
test_ds = test_ds.skip(6)
print(len(test_ds))

#------ in python function ---- 
# --------------- Custom Function for Dataset Partitioning ------------------
# A custom Python function named 'get_dataset_partitions_tf' is defined to partition 
# the dataset. It takes the dataset and desired split ratios as input and returns 
# separate datasets for training, validation, and testing.
#watch shuffle size video to get 'the idea
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split *ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds

# call the function

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

print(len(train_ds))

print(len(test_ds))

print(len(val_ds))


# ------------------ Cache, Shuffle, and Prefetch -----------------
# Finally, the training, validation, and test datasets are cached, shuffled, and prefetched 
# to optimize data loading during model training.
# video on tensorflow data input pipeline concept behind cashing and pre-fetching

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# above datasets are kind of optimized   Tutorial 26
##########################################################

# -------------- Resize and Rescale -----------------
# This part defines a sequential model using 'tf.keras.Sequential'. It consists of two layers:
# 'tf.keras.layers.Resizing' and 'tf.keras.layers.Rescaling'. The Resizing layer resizes the 
# images to the specified dimensions (IMAGE_SIZE x IMAGE_SIZE), while the 'Rescaling' layer 
# normalizes the pixel values to the range [0,1] by dividing each pixel value by 255.

resize_and_rescale= tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    tf.keras.layers.Rescaling(1.0/255)])

# tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
# tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)])



# preprocessing
#previously you got RGB number 0-255if we devide that number from 255 we get numbers inbetween 0-1
# resizing take careof when the image size is not 256x256
# resize_and_rescale = tf.keras.Sequential([
#     tf.keras.layers.experimental.processing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
#     tf.keras.layers.experimental.processing.Rescaling(1.0/255)
# ])

# -------------- Data Augmentation -------------------
# Another sequential model is defined for data augmentation, which includes two layers: 
# 'tf.keras.layers.RandomFlip' and tf.keras.layers.RandomRotation'. These layers introduce 
# random flips (both horizontally and vertically) and random rotations (up to 20 degrees) to 
# the images, thereby increasing the diversity of the training data and improving the model's generalization.
# # data augmentation check video. when image rotation, horizontal flip, contrast, zoom 

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vartical"),
    tf.keras.layers.RandomRotation(0.2)
    
    # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vartical"),
    # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

# next video: buld a model and train it  
# This video !. We loaded our data into tensorflow dataset, some visualization, some split, some preprocessing
# create layer for preprocessing


   ######
   # This issue has been automatically marked as stale because it has no recent activity. It will be closed if no further activity occurs. 
   #####

# Video 3 : build the model using CNN-Convolutional famouse NN architecture for image classification problems
# watch CNN 23 video
# CNN --> convolutional and pulling layers// and then dense layer where you flattern it
# check tensorflow Conv2D(filters/layers(filter to detect eye,nose,hands, ect), kernal size,)

# ------------- Model Architecture -------------------
# The main architecture of the convolutional neural network (CNN) model is defined. 
# It starts with the resized and rescaled images, followed by data augmentation. 
# Then, a series of convolutional ('layers.Conv2D') and max-pooling ('layers.MaxPooling2D') 
# layers are stacked to extract features from the images. The number of filters and
# kernel sizes for the convolutional layers are specified. The final layers include 
# flattening the feature maps ('tf.keras.layers.Flatten') and two dense layers ('layers.Dense') 
# for classification, with the final layer using the softmax activation function to output 
# class probabilities.

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes =3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    #  fully connected layers in a neural network
    #flttern it -array of neurons and then  hidden layer/densed layer and then the out/classification
    tf.keras.layers.Flatten(),
    layers.Dense(64,activation = 'relu'),
    layers.Dense(n_classes, activation ='softmax'), # softmax will normalise the probability of classes
])

# ------------------ Model Compilation ----------------------
# The model is compiled using the Adam optimizer (optimizer='adam') and 
# sparse categorical cross-entropy loss function 
# (tf.keras.losses.SparseCategoricalCrossentropy). Additionally, accuracy is chosen as the evaluation 
# metric (metrics=['accuracy']).

model.build(input_shape = input_shape)

model.summary()  # before training get the summary it gives parameters to train (trainable parameters)

# in DL we do CNN and then compile using optimizers- adam is the most famous
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']  # accuracy matrics use for training purpose
    )


# ------------------ Model Training ----------------
# The model is trained using the fit method. The training dataset (train_ds) 
# is used for training, with the specified number of epochs (EPOCHS) and batch 
# size (BATCH_SIZE). Validation data (val_ds) is also provided to monitor 
# the model's performance on unseen data during training.
# train the network
#saving in history 
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1, 
    validation_data=val_ds
    )

#Epoch 3/50
# 54/54 ━━━━━━━━━━━━━━━━━━━━ 24s 447ms/step - accuracy: 0.8395(TRINING ACCURACY) - loss: 0.3721 - val_accuracy: 0.9062 - val_loss: 0.2762

# Epoch 48/50
# 54/54 ━━━━━━━━━━━━━━━━━━━━ 24s 443ms/step - accuracy: 0.9979 - loss: 0.0048 - val_accuracy: 0.9740 - val_loss: 0.0791
# Epoch 49/50
# 54/54 ━━━━━━━━━━━━━━━━━━━━ 24s 437ms/step - accuracy: 0.9923 - loss: 0.0205 - val_accuracy: 0.9896 - val_loss: 0.0203
# Epoch 50/50
# 54/54 ━━━━━━━━━━━━━━━━━━━━ 24s 443ms/step - accuracy: 0.9974 - loss: 0.0084 - val_accuracy: 0.9896 - val_loss: 0.0489

# Validation accuracy almost 1

# ---------------- Model Evaluation ------------
# After training, the model is evaluated using the evaluate method with the 
# test dataset (test_ds). This provides insights into the model's performance 
# on unseen data and helps assess its generalization ability.
# Evaluate test run
model.evaluate(test_ds)















