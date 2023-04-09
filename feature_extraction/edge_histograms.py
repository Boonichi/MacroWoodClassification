import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('/home/phannhat/Documents/code/NCKH/dataset/Acrocarpus fraxinifolius/0101.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Sobel filter to detect edges
Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(Gx**2 + Gy**2)
angle = np.arctan2(Gy, Gx) * 180 / np.pi

angle[angle < 0] += 360

T = 100
mag[mag < T] = 0

bins = np.int32(angle / 10)

hist, _ = np.histogram(bins, bins=range(37), weights=mag)

feature = list(hist.astype(np.float32))  
feature.append(gray.shape[0] * gray.shape[1] - np.count_nonzero(mag))

print(len(feature))

