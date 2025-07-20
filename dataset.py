import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class WildfireDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_paths = sorted([
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith(('.jpg', '.png'))
        ])
        self.mask_paths = sorted([
            os.path.join(self.mask_dir, fname)
            for fname in os.listdir(self.mask_dir)
            if fname.endswith(('.jpg', '.png'))
        ])

        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in image and mask counts!"

        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(), 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return image, mask

