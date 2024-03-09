import torch
import torch.nn as nn

# Define the AlexNet class
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
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        print("Input image size:", x.size())
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print("Output size after layer {}: {}".format(idx, x.size()))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        print("Output size after classification:", x.size())
        return x

# Load the image
image_path = "cifar_image.txt"
with open(image_path, "r") as file:
    image_data = file.read().split()

# Convert the data into a torch tensor
image_tensor = torch.tensor([float(pixel) for pixel in image_data])

# Reshape the tensor to match the expected input shape (assuming CIFAR-10 image size)
image_tensor = image_tensor.view(1, 3, 32, 32)

# Define the AlexNet model
model = AlexNet()

# Print input image size
print("Input image size:", image_tensor.size())

# Forward pass through the model
output = model(image_tensor)

