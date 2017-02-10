import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import time
import random
import sys
import os
import dicom
import cv2 as cv
from tensorflow.python.client import timeline
import scipy.ndimage as nd
from skimage import measure, morphology

# Load a patient
DIR_PATH_DATA = "../data/sample"

patients = []
for filename in os.listdir(DIR_PATH_DATA):
    if os.path.isdir(os.path.join(DIR_PATH_DATA, filename)):
        patients.append(filename)


DIR_PATH_LABEL = "../data/stage1"
labels = pd.read_csv(os.path.join(DIR_PATH_LABEL, "stage1_labels.csv"), index_col=0).loc[:, "cancer"].to_dict()


def get_label(patient):
    return labels[patient]

def get_scan(patient):
    slices = [dicom.read_file(os.path.join(DIR_PATH_DATA, patient, s)) for s in os.listdir(os.path.join(DIR_PATH_DATA, patient))]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices

def get_3D_data(patient):
    slices = get_scan(patient)
    image = np.stack([s.pixel_array for s in slices])
    
    image[image == -2000] = 0
    
    for i, s in enumerate(slices):
        intercept, slope = s.RescaleIntercept, s.RescaleSlope
        
        image[i] = intercept + (slope * image[i].astype(np.float64))
        image[i] = image[i].astype(np.int16)
        
    return image

def resample(patient, new_spacing=[1,1,1]):
    scan = get_scan(patient)
    image = get_3D_data(patient)
    
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = nd.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image

# For the sake of testing the network, we'll be using the sample dataset
# For this, we'll use the maximum size of the image
# and PAD any image with -1000 values which is smaller than that

#PS: only the first dimension is different in sample dataset
#which is not the case in actual dataset
max_size = (-1, 512, 512)

for idx, patient in enumerate(patients):
    image = get_3D_data(patient)
    print(image.shape)
    if image.shape[0] > max_size[0]:
        max_size = (image.shape[0], max_size[1], max_size[2])

        
print("max size: ", max_size)

def get_padded_image(patient):
    image = get_3D_data(patient)
    
    diff = max_size[0] - image.shape[0]
    
    if diff > 0:
        image = np.vstack((image, np.full(((diff + 1) / 2, image.shape[1], image.shape[2]), fill_value=-1000, dtype=np.int16)))
        image = np.vstack((np.full((diff / 2, image.shape[1], image.shape[2]), fill_value=-1000, dtype=np.int16), image))
        
    return image

#Define constants

glimpseBandwidth = 10

glimpseDepth = 3
glimpseRadius = 20

glimpseZoom = 2

sensorBandwidth = glimpseDepth * (glimpseBandwidth ** 3)

#the size of the network
#How can we make this variable as defined in the research paper
numGlimpses = 6

lr = 5e-1
lrDecayRate = 0.995
lrDecayFreq = 25
momentumValue = 0.9

numEpochs = 200
steps = 2000 #What is this actually?

lowest_number = 1e-10

numClasses = 1 #Simple classification task
InputImageSize = max_size
locationSize = 3
batchSize = 1
biasSize = 1

# Learning Rate
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(lr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

#Inputs
labelsPlaceHolder = tf.placeholder(dtype=tf.int8, 
                                   shape=(batchSize, numClasses), 
                                   name="InputLabels")

imagesPlaceholder = tf.placeholder(dtype=tf.float32, 
                                   shape=(batchSize, InputImageSize[0], InputImageSize[1], InputImageSize[2]), 
                                   name="InputImages")


def weightVariable(shape, name):
    return tf.Variable(tf.random_normal(shape=shape),
                      name=name,
                      trainable=True)


layer_g_h = 128

#Glimpse Encoding
G_g_h_W = weightVariable((sensorBandwidth, layer_g_h), "glimpseEncode_glimpseToHidden_Weights")
G_g_h_B = weightVariable((biasSize, layer_g_h), "glimpseEncode_glimpseToHidden_Bias")

layer_l_h = 128

#Location Encoding
L_l_h_W = weightVariable((locationSize, layer_l_h), "locationEncoding_locationToHidden_Weights")
L_l_h_B = weightVariable((biasSize, layer_l_h), "locationEncoding_locationToHidden_Bias")


g_out_size = 128

#Core Glimpse Network
CGN_gh_gout_W = weightVariable((layer_g_h+layer_l_h, g_out_size), 
                                 "CoreGlimpseNetowrk_GimpseEncodingToGlimpseOutput_Weights")
CGN_lh_gout_W = weightVariable((layer_g_h+layer_l_h, g_out_size), 
                                 "CoreGlimpseNetowrk_locationEncodingToGlimpseOutput_Weights")
                               
CGN_gout_B = weightVariable((1, g_out_size), 
                                 "CoreGlimpseNetowrk_locationEncodingAndGlimpseEncodingToGlipmseOutput_Bias")


netState_size = 128

#Core Network
CN_gout_ns_W = weightVariable((g_out_size, netState_size),
                               "CoreNet_GlimpseToNewState_Weights")
CN_gout_ns_B = weightVariable((1, netState_size),
                             "CoreNet_GlimpseToNewState_Bias")

CN_ns_ns_W = weightVariable((netState_size, netState_size),
                           "CoreNet_OldStateToNewState_Weights")
CN_ns_ns_B = weightVariable((netState_size, netState_size),
                           "CoreNet_OldStateToNewState_Bias")

#Location Network
LN_ns_l_W = weightVariable((netState_size, locationSize),
                          "LocationNet_StateToLocation_Weights")

#BaselineNetwork (For the REINFORCE algorithm on location)
BN_ns_b_W = weightVariable((netState_size, 1),
                          "BaselineNet_StateToBaseline_Weights")
BN_ns_b_B = weightVariable((1, 1),
                          "BaselineNet_StateToBaseline_Bias")

#Action Network (Actual Output network)
AN_ns_a_W = weightVariable((netState_size, numClasses),
                          "ActionNet_StateToAction_Weights")
AN_ns_a_B = weightVariable((1, numClasses),
                          "ActionNet_StateToAction_Bias")

def resizeToGlimpse(image):
    finalShape = tf.constant(glimpseBandwidth, shape=[3])
    currentShape = tf.cast(image.get_shape().as_list(), tf.float32)
    
    zoomFactor = tf.div(finalShape, currentShape)
    return scipy.ndarray.interpolation.zoom(image, zoom=zoomFactor)

def getGlipmse(img, location, radius):
    endPoint = tf.constant(radius*2, shape=location.get_shape())
    
    location = tf.maximum(0, location - radius)
    endPoint = tf.minimum(InputImageSize, location + endPoint)
        
    slicedImage = img[location[0]:endPoint[0], 
                      location[1]:endPoint[1], 
                      location[2]:endPoint[2]]
    
    return resizeToGlimpse(slicedImage)

def glimpseSensor(normalLocation, inputPlaceholder):
    location = tf.round(tf.multiply((normalLocation + 1)/2.0, InputImageSize))
    location = tf.cast(location, tf.int32)
    
    images = tf.reshape(inputPlaceholder, (batchSize, InputImageSize[0], 
                                          InputImageSize[1], 
                                          InputImageSize[2]))
    
    zooms = []
    for k in xrange(batchSize):
        imgZooms = []
        img = images[k]
        
        loc = location[k]
        
        for i in xrange(glimpseDepth):
            radius = int(glimpseRadius * (2 ** i))
            glimpse = getGlipmse(img, loc, radius)
            glimpse = tf.reshape(glimpse, (glimpseBandwidth, glimpseBandwidth, glimpseBandwidth))
            
            imgZooms.append(glimpse)
            
        zooms.append(tf.pack(imgZooms))
        
    zooms = tf.pack(zooms)
    
    return zooms

def glimpseNetwork(location):
    glimpseInput = tf.reshape(glimpseSensor(location, imagesPlaceholder), (batchSize, sensorBandwidths))
    
    glimpseEncode = tf.nn.relu(tf.matmul(glimpseInput, G_g_h_W) + G_g_h_B)
    locationEncode = tf.nn.relu(tf.matmul(location, L_l_h_W) + L_l_h_B)
    
    CGN = tf.nn.relu(tf.matmul(glimpseEncode, CGN_gh_gout_W) 
                     + tf.matmul(locationEncode, CGN_lh_gout_W)
                     + CGN_gout_B)
    
    return CGN

baselines = []
meanLocs = []
sampledLocs = []


def getNextGlimpse(output):
    baseline = tf.sigmoid(tf.matmul(output, BN_ns_b_W) + BN_ns_b_B)
    baselines.append(baseline)
    
    mean_loc = tf.matmul(output, LN_ns_l_W)
    mean_loc = tf.stop_gradient(mean_loc)
    meanLocs.append(mean_loc)
    
    sample_loc = tf.maximum(-1.0, tf.minimum(1.0, mean_loc + tf.random_normal(mean_loc.get_shape(), 0, 0.1)))
    sample_loc = tf.stop_gradient(sample_loc)
    sampledLocs.append(sample_loc)
    
    return glimpseNetwork(sample_loc)

def CoreNetwork():
    initialLocation = tf.random_uniform((batchSize, 3), minval=-1, maxval=+1)
    meanLocs.append(initialLocation)
    
    sample_loc = tf.tanh(initialLocation + tf.random_normal(initialLocation.get_shape(), 0, 0.1))
    sampledLocs.append(sample_loc)
    
    glimpse = glimpseNetwork(sample_loc)
    
    outputs = [0] * numGlimpses
    
    netState = tf.zeros((batchSize, netState_size))
    for net in xrange(numGlimpses):
        netState_old = netState
        
        netState = tf.nn.relu((tf.matmul(netState_old, CN_ns_ns_W) + CN_ns_ns_B) 
                              + (tf.matmul(glimpse, CN_gout_ns_W) + CN_gout_ns_B))
        
        outputs[net] = netState
        
        if net != numGlimpses-1:
            glimpse = getNextGlimpse(netState)
        else:
            baseline = tf.sigmoid(tf.matmul(netState, BN_ns_b_W) + BN_ns_b_B)
            baselines.append(baseline)
        
    return outputs


outputs = CoreNetwork()