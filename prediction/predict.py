import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

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
        relu_output = self.features[0](x)
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)  # First Convolutional + ReLU + MaxPool

        # Process through second layers
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)  # Second Convolutional + ReLU + MaxPool

        # Continue with the rest of the layers
        for i in range(6, len(self.features)):
            x = self.features[i](x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, relu_output


# Define transformations for input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize input image to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
])

# Load the trained model
model = AlexNet()
model.load_state_dict(torch.load('alexnet_cifar10.pth'))
model.eval()  # Set model to evaluation mode

# Function to predict class label for a single image
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient computation for inference
        output, relu_output = model(image_tensor)
        _, predicted = output.max(1)
        return predicted.item(), relu_output

# Example usage
image_path = 'image_2.png'  # Path to your image
predicted_class, relu_output = predict_image(image_path)
print(f'Predicted class: {predicted_class}')

# Print the output after the first convolution and ReLU layer
print('Output after first conv and ReLU:', relu_output)

