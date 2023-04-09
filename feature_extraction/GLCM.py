import skimage.feature
import numpy as np
import scipy.stats

def GLCM(image, distance, angle, level):
    glcm = skimage.feature.graycomatrix(image, distance, angle, levels=level)
    energy = skimage.feature.graycoprops(glcm, 'energy')
    contrast = skimage.feature.graycoprops(glcm, 'contrast')
    entropy = skimage.feature.graycoprops(glcm, 'entropy')
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
    correlation = skimage.feature.graycoprops(glcm, 'correlation')
    maximum_probability = np.max(glcm)
    third_moment = scipy.stats.moment(np.reshape(glcm, (len(glcm)**2)), moment=3)
    return [energy, contrast, entropy, homogeneity, maximum_probability, correlation, third_moment]

