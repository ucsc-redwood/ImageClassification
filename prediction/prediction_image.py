import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
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
        self.classifier = nn.Linear(256 * 7 * 7, num_classes)  # Adjusted for the image size after feature extraction

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the model and load the trained weights
model = AlexNet(num_classes=10)
model.load_state_dict(torch.load('cifar_net.pth'))
model.eval()
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),  # Ensuring the image is 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load and transform the image
image = Image.open('original_image.png')
image = transform(image).unsqueeze(0)  # Add batch dimension
image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Make a prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)

# Output the model's prediction
print(f'Predicted: {predicted[0].item()}')

