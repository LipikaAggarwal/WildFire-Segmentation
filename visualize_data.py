import os
import cv2
import matplotlib.pyplot as plt

# Define paths
fire_dir = 'data/fire/images/fire'
not_fire_dir = 'data/fire/images/not_fire'
masks_dir = 'data/fire/masks'

# Load a few sample images from each category
def load_images_from_folder(folder, max_images=5):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames[:max_images]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img_path, img))
    return images

def show_images_with_masks(images, mask_folder):
    for img_path, img in images:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_folder, filename)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Mask' if mask is not None else 'No Mask Found')
        if mask is not None:
            plt.imshow(mask, cmap='gray')
        else:
            plt.text(0.5, 0.5, 'No Mask', ha='center')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Load and display
print("Showing fire images and their masks...")
fire_images = load_images_from_folder(fire_dir)
show_images_with_masks(fire_images, masks_dir)

print("Showing not_fire images and their masks...")
not_fire_images = load_images_from_folder(not_fire_dir)
show_images_with_masks(not_fire_images, masks_dir)
