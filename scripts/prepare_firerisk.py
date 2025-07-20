import os
import shutil
from PIL import Image
from glob import glob

src_img = "data/fire_risk/FireRisk-main/images"
src_label = "data/fire_risk/FireRisk-main/labels"

for i, img_path in enumerate(glob(src_img + "/*.png")[:200]):
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    img.save(f"data/images/{i:05}.png")

    label_path = os.path.join(src_label, os.path.basename(img_path))
    mask = Image.open(label_path).convert("L").resize((256, 256))
    mask.save(f"data/masks/{i:05}.png")
