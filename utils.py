import cv2
import os
import numpy as np

def get_images(real_path, gen_path):
    im1 = [cv2.imread(os.path.join(real_path, img)) for img in os.listdir(real_path) if img.split(".")[0].isnumeric()]
    im2 = [cv2.imread(os.path.join(gen_path, img)) for img in os.listdir(gen_path) if img.split(".")[0].isnumeric()]
    return np.asarray(im1).astype('float32'), np.asarray(im2).astype('float32')