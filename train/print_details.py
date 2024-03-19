import torch
import torch.nn as nn

# Define the modified AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjusted for 32x32 input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
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
        # Adjusted input size for the linear classifier
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)  # Assuming two max pool layers reducing size to 4x4

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the pruned AlexNet model
model_path = 'pruned_alexnet_cifar10_0.5.pth'
pruned_model = AlexNet()
pruned_model.load_state_dict(torch.load(model_path))

# Print the model architecture
print(pruned_model)

# Print the shape of each layer
for name, param in pruned_model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}")

