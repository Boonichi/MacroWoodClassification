import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def LBP(image, P, R):
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    # Tính toán LBP cho mỗi điểm ảnh
    for i in range(R, rows - R):
        for j in range(R, cols - R):
            center = image[i, j]
            code = 0
            for k in range(P):
                angle = 2 * np.pi * k / P
                x = int(round(i + R * np.cos(angle)))
                y = int(round(j - R * np.sin(angle)))
                if image[x, y] >= center:
                    code |= 1 << (P - 1 - k)
            result[i, j] = code
    
    return result

# Đọc ảnh từ file
image_path = "test.jpg"

image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
gray_image = transform(image)[0]

gray_image = np.array(gray_image)


# Tính toán LBP với P=8, R=1
lbp_image = LBP(gray_image, 8, 1)

print(lbp_image)
img = Image.fromarray(lbp_image)
img.show()