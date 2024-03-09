# ImageClassification

## Structure of Alexnet Cifar-10
```python
# Define the AlexNet model architecture using PyTorch

import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Define the convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Define the fully connected layers
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Forward pass through the network
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

## Input and Output shapes for each layer
```

Input for First_Conv2d ([1, 3, 32, 32])
Output for First_Conv2d ([1, 64, 8, 8])

Input for First_ReLU ([1, 64, 8, 8])
Output for First_ReLU ([1, 64, 8, 8])

Input for First Max_Pool2d ([1, 64, 8, 8])
Output for First Max_Pool2d ([1, 64, 4, 4])

Input for Second_Conv2d ([1, 64, 4, 4])
Output for Second_conv2d ([1, 192, 4, 4])

Input for Second_ReLU ([1, 192, 4, 4])
Output for Second_ReLU ([1, 192, 4, 4])

Input for Second_Maxpool ([1, 192, 4, 4])
Output for Second_Maxpool ([1, 192, 2, 2])

Input for Third_Conv2d([1, 192, 2, 2])
OutPut for Third_Conv2d([1, 384, 2, 2])

Input for Third_ReLU ([1, 384, 2, 2])
Output for Third_Relu ([1, 384, 2, 2])

Input for Fourth_Conv2d ([1, 384, 2, 2])
Output for Fourth_Conv2d ([1, 256, 2, 2])

Input for Fourth_ReLU ([1, 256, 2, 2])
OutPut for Fourth_ReLU ([1, 256, 2, 2])

Input for Fifth_Conv2d ([1, 256, 2, 2])
Output for Fifth_Conv2d ([1, 256, 2, 2])

Input for Fifth_ReLU ([1, 256, 2, 2])
Output for Fifth_ReLU ([1, 256, 2, 2])

Input for Third_maxPool  ([1, 256, 2, 2])
OutPut for Third_maxpool ([1, 256, 1, 1])

Output size after classification: torch.Size([1, 10])
```

## Output from Cifar10.cpp

```
First_conv2d Input: 32x32x3 First_conv2d Output: 8x8x64
First_ReLU Input: 8x8x64 First_ReLU Output: 8x8x64
First_maxpool2d Input: 8x8x64 First_maxpool2d Output: 4x4x64
Second_conv2d Input: 4x4x64 Second_conv2d Output: 4x4x192
Second_ReLU Input: 4x4x192 Second_ReLU Output: 4x4x192
Second_maxpool2d Input: 4x4x192 Second_maxpool2d Output: 2x2x192
Third_conv2d Input: 2x2x192 Third_conv2d Output: 2x2x384
Third_ReLU Input: 2x2x384 Third_ReLU Output: 2x2x384
Fourth_conv2d Input: 2x2x384 Fourth_conv2d Output: 2x2x256
Fourth_ReLU Input: 2x2x256 Fourth_ReLU Output: 2x2x256
Fifth_conv2d Input: 2x2x256 Fifth_conv2d Output: 2x2x256
Fifth_ReLU Input: 2x2x256 Fifth_ReLU Output: 2x2x256
Third_maxpool2d Input: 2x2x256 Third_maxpool2d Output: 1x1x256
view_flattend Input: 1x1x256 view_flattend Output: 1x256
Max value: 0.238324 Image class: 4
```
