import torch
import torchvision
import os
import numpy as np
from torchvision import transforms

# Function to create directories if they don't exist
def create_directories(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a directory to save images
save_dir = './cifar10_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create directories for each class
class_names = train_dataset.classes
class_dirs = [os.path.join(save_dir, class_name) for class_name in class_names]
create_directories(class_dirs)

# Function to save images of each class
def save_images_by_class(dataset, num_images_per_class, class_dirs):
    class_counts = {class_dir: 0 for class_dir in class_dirs}
    for image, label in dataset:
        class_dir = class_dirs[label]
        image_name = f"{class_counts[class_dir]}.png"
        image_path = os.path.join(class_dir, image_name)
        torchvision.utils.save_image(image, image_path)
        class_counts[class_dir] += 1
        if all(count >= num_images_per_class for count in class_counts.values()):
            break

# Save 100 images of each class
num_images_per_class = 100
save_images_by_class(train_dataset, num_images_per_class, class_dirs)

print("Images saved successfully.")

