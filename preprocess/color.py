import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import color
import cv2
import numpy as np
from scipy.stats import kurtosis, skew


def RGB_color_extracting(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    grid_size = 3
    height, width, channels = img.shape
    grid_h, grid_w = int(height/grid_size), int(width/grid_size)

    features = np.zeros((grid_size, grid_size, 3, 3))

    for i in range(grid_size):
        for j in range(grid_size):
            roi = img[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]

            for c in range(channels):
                mean = np.mean(roi[:,:,c])
                var = np.var(roi[:,:,c])
                skewness = np.mean(np.power(roi[:,:,c]-mean, 3))/np.power(np.sqrt(var), 3)

                features[i, j, c, 0] = mean
                features[i, j, c, 1] = var
                features[i, j, c, 2] = skewness

    feature_vector = features.reshape(grid_size * grid_size * 3 * 3)
    return feature_vector

def LAB_color_extracting(img_path):
    img = cv2.imread(img_path)

    grid_size = 3
    height, width, channels = img.shape
    grid_h, grid_w = int(height/grid_size), int(width/grid_size)

    features = np.zeros((grid_size, grid_size, 3, 3))

    for i in range(grid_size):
        for j in range(grid_size):
            roi = img[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]

            for c in range(channels):
                mean = np.mean(roi[:,:,c])
                var = np.var(roi[:,:,c])
                skewness = np.mean(np.power(roi[:,:,c]-mean, 3))/np.power(np.sqrt(var), 3)

                features[i, j, c, 0] = mean
                features[i, j, c, 1] = var
                features[i, j, c, 2] = skewness

    feature_vector = features.reshape(grid_size * grid_size * 3 * 3)
    return feature_vector

def GSL_color_extracting(img_path):
    img = cv2.imread(img_path)

    region1 = img[:200]
    region2 = img[200:]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cieluv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    channels = [("G", region1[:, :, 1]), ("S", hsv[:, :, 1]), ("L", cieluv[:, :, 0])]

    features = np.zeros((2, 3, 3))

    for i in range(2):
        for j, col in enumerate(channels):
            channel_name, channel_data = col

            if i == 0:
                data = channel_data[:200]
            else:
                data = channel_data[200:]

            features[i, j, 0] = np.mean(data)
            features[i, j, 1] = kurtosis(data.flatten())  
            features[i, j, 2] = skew(data.flatten())

    feature_vector = features.flatten()
    return(feature_vector)

img_path = 'test.jpg'

out_img = Image.fromarray(LAB_color_extracting(img_path), 'RGB')

out_img.show()
