import cv2
import numpy as np
import scipy.stats

IMAGE_PATH = "test.jgp"

def scale(img, size):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) 
    new_img *= size
    return new_img

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

def gabor_filter(img, gabor_bank):
    res = []
    for kernel in gabor_bank:
        res.append(apply_sliding_window_on_3_channels(img, kernel))
    return res

def feature_gabor_filter(gabor_filter):
    mean = np.mean(gabor_filter)
    variance = scipy.stats.moment(np.reshape(gabor_filter, (len(gabor_filter)**2)), moment=2)
    skewness = scipy.stats.skew(gabor_filter)
    return [mean, variance, skewness]
