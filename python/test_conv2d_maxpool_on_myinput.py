import torch
import torch.nn as nn
import numpy as np

def load_tensor_from_txt(file_name, dtype=np.float32):
    return torch.tensor(np.loadtxt(file_name, dtype=dtype))

# Load the flattened image and reshape it
flattened_image = load_tensor_from_txt('flattened_image_tensor.txt')[:3072]  # Load and trim the image data
image_size = int((flattened_image.shape[0] // 3) ** 0.5)  # Calculate the image size
input_image = flattened_image.reshape(1, 3, image_size, image_size)

# Load the weights and biases
weights = load_tensor_from_txt('features_0_weight.txt').reshape(64, 3, 3, 3)
bias = load_tensor_from_txt('features_0_bias.txt')

# Ensure the bias tensor has the correct shape (64,)
assert bias.shape[0] == 64

# Define the Conv2d layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

# Set the weights and bias of the convolution layer
conv_layer.weight.data = weights
conv_layer.bias.data = bias

# Perform convolution
output = conv_layer(input_image)

# Define the ReLU activation function
relu = nn.ReLU()

# Apply ReLU activation
output_relu = relu(output)

# Define the MaxPool2d layer
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Apply MaxPool2d
output_max_pool = max_pool(output_relu)

# Print the output after max pooling
print("Output after MaxPool2d:")
print(output_max_pool.squeeze().detach().numpy())

