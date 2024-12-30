import torch

import torch.nn as nn
import torch.nn.functional as F

class VGG16Network(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(VGG16Network, self).__init__()
        # Define the VGG network architecture
        self.features = nn.Sequential(
            self._convrelublock(1, 64, 2),
            self._convrelublock(64, 128, 2),
            self._convrelublock(128, 256, 3),
            self._convrelublock(256, 512, 3),
            self._convrelublock(512, 512, 3)
        )
            # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 4 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),)
    
    def _convrelublock(self, in_channels, out_channels, layers):
        block = []
        for _ in range(layers):
            block.append(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1))
            block.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        block.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*block)
      

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        

# Example usage
if __name__ == "__main__":
    input_dim = 132 * 175 * 48
    output_dim = len(cfg.data.emotion_idx)
    model = VGG16Network()
    print(model)