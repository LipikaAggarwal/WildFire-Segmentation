import segmentation_models_pytorch as smp
import torch.nn as nn

class PretrainedUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'):
        super(PretrainedUNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)
