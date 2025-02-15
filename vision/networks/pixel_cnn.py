import torch
from torch import nn
import numpy as np
from pixelcnn.layers import *


class PixelCNN(nn.Module):
    """
    Conditional PixelCNN using gated convolutional layers
    
    n_channels: channels for each convolutional layer
    n_layers: number of intermediate convolutional layers
    """
    def __init__(self, n_channels=32, n_layers=7,img_shape=(3,32,32)):
        super(PixelCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(CondGatedMaskedConv2d('A', 1, n_channels,
                                                 7, 1, 3, bias=False))
        self.layers.append(nn.BatchNorm2d(n_channels))
        
        for i in range(1, n_layers+1):
            self.layers.append(CondGatedMaskedConv2d('B', n_channels,
                                                     n_channels, 7, 1, 3,
                                                     bias=False))
            self.layers.append(nn.BatchNorm2d(n_channels))

        # map to 256 output channels
        self.layers.append(nn.Conv2d(n_channels, 256, 1))
        self.img_shape = img_shape
        self.n_channels = n_channels
    
    def forward(self, x, h):
        [c,h,w] = self.img_shape
        img = torch.zeros(self.img_shape, dtype=torch.long)

        for layer in self.layers:
            # only pass conditional input to the CondGatedMaskedConv2d layers
            if isinstance(layer, CondGatedMaskedConv2d):
                out = layer(x, h)
            else:
                out = layer(x)


        probs = nn.functiona.softmax(out[:, :, c, h, w], dim=-1)
        img[:, c, h, w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img
                              

class LabelNet(nn.Module):
    """
    Linear layer to map from one-hot-encoded label to 28x28
    """
    def __init__(self, input_shape=10, output_shape=(28,28)):
        super(LabelNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.linear = nn.Linear(10, np.prod(output_shape))

    def forward(self, h):
        return self.linear(h).view(-1, 1, *self.output_shape)
