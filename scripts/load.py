import os
import cv2
import matplotlib.pyplot as plt
from glob import glob

# Paths
fire_path = 'data/fire/images/fire'
not_fire_path = 'data/fire/images/not_fire'
masks_path = 'data/fire/masks'

# Get all image paths
fire_images = sorted(glob(os.path.join(fire_path, '*')))
not_fire_images = sorted(glob(os.path.join(not_fire_path, '*')))
mask_images = sorted(glob(os.path.join(masks_path, '*')))

# Print dataset info
print(f"Total fire images: {len(fire_images)}")
print(f"Total not fire images: {len(not_fire_images)}")
print(f"Total masks: {len(mask_images)}")

# Function to load and resize image
def load_image(path, size=(256, 256), gray=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    return img

# Visualize few samples
for i in range(3):
    fire_img = load_image(fire_images[i])
    mask_img = load_image(mask_images[i], gray=True)

    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(fire_img, cv2.COLOR_BGR2RGB))
    plt.title('Fire Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_img, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
