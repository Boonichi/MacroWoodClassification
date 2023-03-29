import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('test.jpg', 0)

# Apply Sobel filter to detect edges
dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
mag = np.sqrt(dx**2 + dy**2)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Compute edge histograms
hist, bins = np.histogram(mag, bins=8, range=(0, 255))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

# Visualize histogram
plt.bar(bins[:-1], hist, width = 10)
plt.xlim(0, 255)
plt.show()
