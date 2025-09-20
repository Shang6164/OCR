import torch
import torch.nn as nn
import torchvision.models as models

class CenterNetMobileNetV2(nn.Module):
    def __init__(self, num_classes=1):
        super(CenterNetMobileNetV2, self).__init__()
        
        # Load pre-trained MobileNetV2 backbone
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet_v2.features
        
        # Decoder layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(1280, 96, kernel_size=1)
        self.conv2 = nn.Conv2d(96 + 32, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64 + 24, num_classes, kernel_size=1)

    def forward(self, x):
        # Extract features
        c3_output = self.features[:7](x)  # Output of layer 7
        c4_output = self.features[7:14](x)  # Output of layer 14
        c5_output = self.features[14:](x)  # Output of final layer

        # Decoder
        x = self.upsample1(c5_output)
        x = torch.cat([x, c4_output], dim=1)
        x = self.conv1(x)

        x = self.upsample2(x)
        x = torch.cat([x, c3_output], dim=1)
        x = self.conv2(x)

        x = self.conv3(x)
        return x

if __name__ == "__main__":
    model = CenterNetMobileNetV2(num_classes=1)
    print(model)