import skimage.feature
import numpy as np
import scipy.stats
import cv2
import skimage

def GLCM(image, distance = [1], angle = [45], level = 256):
    glcm = skimage.feature.graycomatrix(image.astype(int), distance, angle, levels=level)
    energy = skimage.feature.graycoprops(glcm, 'energy')
    contrast = skimage.feature.graycoprops(glcm, 'contrast')
    entropy = skimage.feature.graycoprops(glcm, 'entropy')
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
    correlation = skimage.feature.graycoprops(glcm, 'correlation')
    maximum_probability = np.max(glcm)
    third_moment = scipy.stats.moment(np.reshape(glcm, (len(glcm)**2)), moment=3)
    return [energy, contrast, entropy, homogeneity, maximum_probability, correlation, third_moment]

img = cv2.imread("test.jpg")
gray_img = skimage.color.rgb2gray(img)
print(GLCM(gray_img))