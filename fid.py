import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from scipy.linalg import sqrtm

import cv2
import argparse
import matplotlib.pyplot as plt
from keras.utils import generic_utils
import time
try: import cPickle as pickle 
except: import pickle
    
class FID:
    #Create Inception V3 Model
    def __init__(self, input_shape=(299,299,3)):
        self.input_shape = input_shape
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)

    #Scale Images
    def scale_images(self, images):
        return np.asarray([cv2.resize(im, (self.input_shape[0], self.input_shape[1]), interpolation = cv2.INTER_CUBIC) for im in images])
    
    # calculate frechet inception distance
    def calculate_fid(self, images1, images2):
        # calculate activations
        act1 = self.model.predict(images1)
        act2 = self.model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean): covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
 
    def process_images(self, images1, images2):
        images1 = self.scale_images(images1)
        images2 = self.scale_images(images2)
        
        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)
        
        return images1, images2		
    
    def find_fid(self, images1, images2):
        images1, images2 = self.process_images(images1, images2)
        fid = self.calculate_fid(images1, images2)
        return fid

