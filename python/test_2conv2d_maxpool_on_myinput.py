import torch
import torch.nn as nn
import numpy as np

def load_tensor_from_txt(file_name, dtype=np.float32):
    return torch.tensor(np.loadtxt(file_name, dtype=dtype))

# Load the flattened image and reshape it
flattened_image = load_tensor_from_txt('flattened_image_tensor.txt')[:3072]
image_size = int((flattened_image.shape[0] // 3) ** 0.5)
input_image = flattened_image.reshape(1, 3, image_size, image_size)

# Load weights and biases for the first convolutional layer
weights1 = load_tensor_from_txt('features_0_weight.txt').reshape(64, 3, 3, 3)
bias1 = load_tensor_from_txt('features_0_bias.txt')

# Define the first Conv2d layer
conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
conv_layer1.weight.data = weights1
conv_layer1.bias.data = bias1

# Apply the first convolution
output1 = conv_layer1(input_image)

# Define and apply the ReLU activation
relu1 = nn.ReLU()
output_relu1 = relu1(output1)

# Define and apply the first MaxPool2d layer
max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
output_max_pool1 = max_pool1(output_relu1)

# Load weights and biases for the second convolutional layer
weights2 = load_tensor_from_txt('features_3_weight.txt').reshape(192, 64, 3, 3)
bias2 = load_tensor_from_txt('features_3_bias.txt').reshape(192)

# Define the second Conv2d layer
conv_layer2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=True)
conv_layer2.weight.data = weights2
conv_layer2.bias.data = bias2

# Apply the second convolution
output2 = conv_layer2(output_max_pool1)

# Apply the second ReLU activation
relu2 = nn.ReLU()
output_relu2 = relu2(output2)

# Print the specified channel outputs after the second ReLU
channels_to_print = [0, 49, 99, 191]
for channel in channels_to_print:
    print(f"Output for channel {channel + 1} after the second ReLU:")
    print(output_relu2[0, channel, :, :].squeeze().detach().numpy())
    print("\n")

