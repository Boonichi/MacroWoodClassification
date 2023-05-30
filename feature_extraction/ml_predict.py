import torch
from PIL import Image
import cv2
import numpy as np
from scipy.stats import kurtosis, skew
import os

from skimage.color import rgb2gray
from skimage import io
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.metrics import accuracy_score


#Color extracting
def RGB_color_extracting(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    grid_size = 3
    height, width, channels = img.shape
    grid_h, grid_w = int(height/grid_size), int(width/grid_size)

    features = np.zeros((grid_size, grid_size, 3, 3))

    for i in range(grid_size):
        for j in range(grid_size):
            roi = img[i*grid_h:(i + 1)*grid_h, j*grid_w:(j + 1)*grid_w]

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

def color_extracting(img_path):
    RGB = RGB_color_extracting(img_path)
    LAB = LAB_color_extracting(img_path)
    return np.concatenate([RGB, LAB])


#GLCM
def GLCM(img_path):
    image_rgb = io.imread(img_path)
    image = img_as_ubyte(rgb2gray(image_rgb))
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, np.pi*3/4], levels=256)
    energy = graycoprops(glcm, 'energy')
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    correlation = graycoprops(glcm, 'correlation')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    return np.array([energy, contrast, homogeneity, correlation, dissimilarity]).flatten()


#Gabor filter
def apply_sliding_window_on_3_channels(img, kernel):
    layer_blue = cv2.filter2D(src=img[:,:,0], ddepth=-1, kernel=kernel)
    layer_green = cv2.filter2D(src=img[:,:,1], ddepth=-1, kernel=kernel)
    layer_red = cv2.filter2D(src=img[:,:,2], ddepth=-1, kernel=kernel)    
    
    new_img = np.zeros(list(layer_blue.shape) + [3])
    new_img[:,:,0], new_img[:,:,1], new_img[:,:,2] = layer_blue, layer_green, layer_red
    return new_img

def generate_gabor_bank(num_kernels, ksize=(64, 64), sigma=3, lambd=6, gamma=0.25, psi=0):
    bank = []
    theta = 0
    step = np.pi / num_kernels
    for idx in range(num_kernels):
        theta = idx * step
        kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
        bank.append(kernel)
    return bank

def feature_gabor_filter(gabor_filter):
    mean = np.mean(gabor_filter)
    variance = np.std(gabor_filter)
    return [mean, variance]

def gabor_extraction(img_path):
    img = cv2.imread(img_path)
    gabor_bank = generate_gabor_bank(num_kernels=8)
    feature = []
    for kernel in gabor_bank:
        feature.append(feature_gabor_filter(apply_sliding_window_on_3_channels(img, kernel)))
    return np.array(feature).flatten()


#LBP 
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def LBP(img_path):
    img_bgr = cv2.imread(img_path, 1)
    height, width, _ = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
   
    img_lbp = np.zeros((height, width), np.uint8)
   
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp.flatten()


#Edge histograms
def edge_extraction(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to detect edges
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi

    angle[angle < 0] += 360

    T = 20
    mag[mag < T] = 0

    bins = np.int32(angle / 10)

    hist, _ = np.histogram(bins, bins=range(37), weights=mag)

    feature = list(hist.astype(np.float32))  
    feature.append(gray.shape[0] * gray.shape[1] - np.count_nonzero(mag))
    return feature



X_train = []
y_train = []
X_test = []
y_test = []
for species in os.listdir('./fold_0/train/'):
    for img_name in os.listdir('./fold_0/train/' + species):
        img_path = './fold_0/train/' + species + '/' + img_name
        color = color_extracting(img_path)
        glcm = GLCM(img_path)
        gabor = gabor_extraction(img_path)
        edge = edge_extraction(img_path)
        X_train.append(np.concatenate([color, glcm, gabor, edge]))
        y_train.append(species)
    
for species in os.listdir('./fold_0/test/'):
    for img_name in os.listdir('./fold_0/test/' + species):
        img_path = './fold_0/train/' + species + '/' + img_name
        color = color_extracting(img_path)
        glcm = GLCM(img_path)
        gabor = gabor_extraction(img_path)
        edge = edge_extraction(img_path)
        X_test.append(np.concatenate([color, glcm, gabor, edge]))
        y_test.append(species)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
