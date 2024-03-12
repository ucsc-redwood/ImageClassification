import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

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
        self.classifier = nn.Linear(256 * 7 * 7, num_classes)

    def forward(self, x):
        print("Value of x before features: ", x.shape)
        for layer in self.features:
            x = layer(x)
            print("Value of x after", layer, ":", x.shape)
        x = torch.flatten(x, 1)
        print("Value of x after flatten:", x.shape)
        x = self.classifier(x)
        print("Value of x after linear:", x.shape)
        return x

# Initialize the model and load the trained weights
model = AlexNet(num_classes=10)
model.load_state_dict(torch.load('cifar_net.pth'))
model.eval()
model.cuda()  # Move the model to GPU if available

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the CIFAR-10 test dataset
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# Fetch a single image and label
images, labels = next(iter(testloader))

# Save the original image
save_image(images, 'original_image.png')

images = images.cuda()  # Move the images to GPU if available

# Make a prediction
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

# Output the model's prediction
print(f'Predicted: {predicted[0].item()}, Actual: {labels[0]}')

