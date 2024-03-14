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

# Define and apply the second MaxPool2d layer
max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
output_max_pool2 = max_pool2(output_relu2)

# Load weights and biases for the third convolutional layer
weights3 = load_tensor_from_txt('features_6_weight.txt').reshape(384, 192, 3, 3)
bias3 = load_tensor_from_txt('features_6_bias.txt').reshape(384)

# Define the third layer
conv_layer3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True)
conv_layer3.weight.data = weights3
conv_layer3.bias.data = bias3

# Apply the third convolution
output3 = conv_layer3(output_max_pool2)

# Apply the third ReLU activation
relu3 = nn.ReLU()
output_relu3 = relu3(output3)

# Load weights and biases for the fourth convolutional layer
weights4 = load_tensor_from_txt('features_8_weight.txt').reshape(256, 384, 3, 3)
bias4 = load_tensor_from_txt('features_8_bias.txt').reshape(256)

# Define the fourth layer
conv_layer4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
conv_layer4.weight.data = weights4
conv_layer4.bias.data = bias4

# Apply the fourth convolution
output4 = conv_layer4(output_relu3)

# Apply the fourth ReLU activation
relu4 = nn.ReLU()
output_relu4 = relu4(output4)

# Load weights and biases for the fifth convolutional layer
weights5 = load_tensor_from_txt('features_10_weight.txt').reshape(256, 256, 3, 3)
bias5 = load_tensor_from_txt('features_10_bias.txt').reshape(256)

# Define the fifth layer
conv_layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
conv_layer5.weight.data = weights5
conv_layer5.bias.data = bias5

# Apply the fifth convolution
output5 = conv_layer5(output_relu4)

# Apply the fifth ReLU activation
relu5 = nn.ReLU()
output_relu5 = relu5(output5)

# Define and apply the third MaxPool2d layer
max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
output_max_pool3 = max_pool3(output_relu5)

flatten =  output_max_pool3.view(output_max_pool3.size(0), -1)

# Load weights and biases for the linear layer
linear_weights = load_tensor_from_txt('classifier_weight.txt').reshape(10, 256 * 4 * 4)
linear_bias = load_tensor_from_txt('classifier_bias.txt').reshape(10)

# Define the linear layer
fc = nn.Linear(256 * 4 * 4, 10)
fc.weight.data = linear_weights
fc.bias.data = linear_bias

# Apply the linear layer
output_linear = fc(flatten)

print("Output of the linear layer:")
print(output_linear)

# Print each element in the flattened tensor
#print("Flattened output elements:")
#for element in flatten[0]:
#    print(element.item())

## Print the specified channel outputs after the second ReLU
#channels_to_print = [0, 50, 255]
#for channel in channels_to_print:
#    print(f"Output for channel {channel + 1} after the second ReLU:")
#    print(output_max_pool3[0, channel, :, :].squeeze().detach().numpy())
#    print("\n")

