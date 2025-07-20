from PIL import Image
import os

def remove_corrupt_images(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print(f"❌ Deleting corrupt file: {file}")
            os.remove(path)

remove_corrupt_images("data/images")
remove_corrupt_images("data/masks")
