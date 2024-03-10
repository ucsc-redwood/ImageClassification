import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def save_images_by_class(dataset, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        class_name = dataset.classes[label]
        
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        filename = os.path.join(class_dir, f'image_{i}.png')
        
        # Save the image
        transform = transforms.ToPILImage()
        image_pil = transform(image)
        image_pil.save(filename)

# Transform to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Save images by class
output_dir = './cifar10_images'
save_images_by_class(trainset, output_dir)

