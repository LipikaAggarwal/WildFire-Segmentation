from model.pretrained_unet import PretrainedUNet
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dataset import WildfireDataset




# Paths
IMAGE_DIR = "data/fire/images/fire"
MASK_DIR = "data/fire/masks"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset and Dataloader
dataset = WildfireDataset(IMAGE_DIR, MASK_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"📦 Dataset contains {len(dataset)} samples.")

# Model
model = PretrainedUNet(
    encoder_name='resnet34',      
    encoder_weights='imagenet',  
    in_channels=3,               
    classes=1,                    
    activation='sigmoid'           
).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).unsqueeze(1) 

        outputs = model(images)
        masks = masks.squeeze(2) 
        loss = criterion(outputs, masks)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"🧠 Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(loader):.4f}")

# Save model
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"Model saved at: {CHECKPOINT_PATH}")
