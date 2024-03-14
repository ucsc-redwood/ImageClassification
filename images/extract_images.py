import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Define transformations for input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize input image to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
])

# Function to convert image to tensor and save the flattened tensor as a text file
def convert_to_tensor(image_path, output_dir):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Flatten the image tensor to a 1D array
    flattened_tensor = image_tensor.view(-1).numpy()
    # Generate output filename
    class_name = os.path.basename(os.path.dirname(image_path))
    output_filename = f'flattened_{class_name}_{os.path.splitext(os.path.basename(image_path))[0]}.txt'
    output_path = os.path.join(output_dir, output_filename)
    # Save the flattened tensor as a text file
    np.savetxt(output_path, flattened_tensor)

# Path to the directory containing the images
image_dir = "/home/riksharm/ImageClassification/images/classes"

# Path to the directory where flattened images will be saved
output_dir = "/home/riksharm/ImageClassification/extract"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through each class directory
for class_name in os.listdir(image_dir):
    class_dir = os.path.join(image_dir, class_name)
    if os.path.isdir(class_dir):
        # Iterate through each image in the class directory
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            # Convert the image to a flattened tensor and save as text file
            convert_to_tensor(image_path, output_dir)

print("Images flattened and saved successfully.")

