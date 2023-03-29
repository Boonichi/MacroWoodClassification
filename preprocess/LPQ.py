import cv2
import numpy as np

def LPQ_features(img, cell_size=16, stride=8):
    # Define LPQ kernel
    lpq_kernel = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 2, 1, 0],
                           [0, 2, -12, 2, 0],
                           [0, 1, 2, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.float32)
    
    # Calculate size of FFT output
    fft_size = cell_size + 4
    
    # Calculate number of cells in x and y direction
    num_cells_x = (img.shape[1] - cell_size) // stride + 1
    num_cells_y = (img.shape[0] - cell_size) // stride + 1
    
    # Create empty LPQ feature vector
    lpq_features = np.zeros((num_cells_y, num_cells_x, 256), dtype=np.float32)
    
    # Iterate over all cells
    for y in range(0, num_cells_y * stride, stride):
        for x in range(0, num_cells_x * stride, stride):
            # Extract cell from image
            cell = img[y:y+cell_size, x:x+cell_size]
            
            # Apply LPQ kernel
            lpq = cv2.filter2D(cell, -1, lpq_kernel)
            
            # Apply short-term Fourier transform
            lpq = lpq.astype(np.float32)
            fft = cv2.dft(lpq, flags=cv2.DFT_COMPLEX_OUTPUT)
            
            # Calculate magnitude and phase spectra
            mag, phase = cv2.cartToPolar(fft[:,:,0], fft[:,:,1])
            
            # Quantize phase information
            bins = np.linspace(-np.pi, np.pi, 256)
            quantized_phase = np.digitize(phase, bins)
            
            # Update LPQ feature vector
            lpq_features[y//stride, x//stride] = np.histogram(quantized_phase, bins=np.arange(257))[0]
    
    # Flatten LPQ feature vector
    lpq_features = lpq_features.reshape(-1)
    
    return lpq_features

import cv2
import torchvision.transforms as transforms
import torchvision
from PIL import Image
img_path = "test.jpg"
img_tensor = torchvision.io.read_image(img_path)

# img = cv2.imread("/home/phannhat/Documents/code/NCKH/dataset/Acrocarpus fraxinifolius/0101.jpg", cv2.IMREAD_GRAYSCALE)
img = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
img = np.array(transform(img).squeeze())
print(LPQ_features(img))