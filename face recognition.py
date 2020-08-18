# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:55:04 2020

@author: yash
"""

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

np.set_printoptions(threshold=2**31)


    

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum((anchor - positive)**2, axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum((anchor - negative)**2, axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    ### END CODE HERE ###
    
    return loss


with  tf.compat.v1.Session() as test:
    tf.random.set_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random.normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))
    
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

def change_size(path_from,path_to):
    # load the image, swap color channels, and resize it to be a fixed
	# 96x96 pixels while ignoring aspect ratio
    image = cv2.imread(path_from)
    #image = image.astype('uint8')
    r, c,_ = image.shape 
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_new = cv2.resize(image, (96,96))

    cv2.imwrite(path_to, image_new) 
    return path_to

#Training Paths
evanspath=change_size("F:\celebrity images\Train\cap.jpg","F:\celebrity images\Test\cap1.jpg")
chrispath=change_size("F:\celebrity images\Test\Original size\ChrisHemsworth4.jpg","F:\celebrity images\Test\ChrisHemsworth4.jpg")
#tompath=change_size("F:\celebrity images\Train\tom.jpg","F:\celebrity images\Train\required size\tom1.jpg")
hollandpath=change_size("F:\celebrity images\Train\holland.jpg","F:\celebrity images\Train\holland2.jpg")
anmolpath=change_size(r"F:\Downloads\anmol.jpg","F:\Downloads\anmol.jpg")
#Testing Paths
evanspathtest=change_size("F:\celebrity images\Test\Original size\ChrisEvans.jpg","F:\celebrity images\Test\Required size\ChrisEvans.jpg")
chrispathtest=change_size("F:\celebrity images\Test\Original size\ChrisHemsworth3.jpg","F:\celebrity images\Test\Required size\ChrisHemsworth3.jpg")
tompathtest=change_size("F:\celebrity images\Test\Original size\TomCruise.jpg","F:\celebrity images\Test\Required size\TomCruise.jpg")
hollandpathtest=change_size("F:\celebrity images\Test\Original size\holland.jpg","F:\celebrity images\Test\Required size\holland.jpg")
anmolpathtest=change_size(r"F:\Downloads\anmol2.jpg","F:\Downloads\anmol2.jpg")


#Making Database
database = {}
database["Chris Evans"] = img_to_encoding(evanspath, FRmodel)
database["Chris Hemsworth"]=img_to_encoding(chrispath, FRmodel)
database["Tom Holland"]=img_to_encoding(hollandpathtest,FRmodel)
database["Anmol Jain"]=img_to_encoding(anmolpath,FRmodel)
#database["Tom Cruise"]=img_to_encoding(tompath, FRmodel)
def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    ### START CODE HERE ###
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist<0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
        
        
    return dist, door_open

#verify(hollandpath, "Tom Holland", database, FRmodel)


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

who_is_it(anmolpathtest, database, FRmodel)

