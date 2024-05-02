import torch
import torch.nn as nn
import numpy as np

def load_and_print_shape(file_path):
    # Load data from text file
    data = np.loadtxt(file_path)

    # Print the shape of the loaded data
    print("Loaded File Shape:", data.shape)

def load_text_data(file_path):
    # Load data from text file
    data = np.loadtxt(file_path)

    # Reshape the data to the required shape (1, 3, 32, 32) using loops
    reshaped_data = np.zeros((1, 3, 32, 32))
    index = 0
    for channel in range(3):
        for i in range(32):
            for j in range(32):
                reshaped_data[0, channel, i, j] = data[index]
                index += 1

    # Print the shape of the reshaped data
    print("Reshaped Data shape:", reshaped_data.shape)
    return reshaped_data

def load_weights_bias(weight_file, bias_file):
    # Load weights and bias from text files
    weights = np.loadtxt(weight_file)
    # Reshape the weights to the correct shape
    weights = weights.reshape((64, 3, 11, 11))
    bias = np.loadtxt(bias_file)

    # Print the expected shape of weights and bias
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)

    # Convert to PyTorch tensors
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    return weights_tensor, bias_tensor

def load_weights_bias1(weight_file, bias_file):
    # Load weights and bias from text files
    weights = np.loadtxt(weight_file)
    # Reshape the weights to the correct shape
    weights = weights.reshape((192, 64, 5, 5))
    bias = np.loadtxt(bias_file)

    # Print the expected shape of weights and bias
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)

    # Convert to PyTorch tensors
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    return weights_tensor, bias_tensor

def convolution_with_input(input_tensor, weights, bias, weights1, bias1):
    # Define the Conv2d model with loaded weights and bias
    class Conv2dModel(nn.Module):
        def __init__(self, weights, bias, weights1, bias1):
            super(Conv2dModel, self).__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=5)
            self.conv.weight = nn.Parameter(weights)
            self.conv.bias = nn.Parameter(bias)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv1 = nn.Conv2d(in_channels=64, out_channels=196, kernel_size=5, stride=1, padding=2)
            self.conv1.weight = nn.Parameter(weights)
            self.conv1.bias = nn.Parameter(bias)
            self.relu1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)  # Adding ReLU activation
            x = self.maxpool(x)  # Adding MaxPool2d layer
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            return x

    # Create an instance of the Conv2d model with loaded weights and bias
    conv_model = Conv2dModel(weights, bias, weights1, bias1)

    # Perform convolution on the input tensor
    output = conv_model(input_tensor)

    # Print output shape
    print("Output shape:", output.shape)

    # Save MaxPool2d output to a text file
    with open("maxpool1_pythonresult.txt", "w") as file:
        for channel in range(output.shape[1]):
            file.write(f"Channel {channel}:\n")
            for i in range(output.shape[2]):
                for j in range(output.shape[3]):
                    file.write(f"{output[0, channel, i, j].item()} ")
                file.write("\n")
            file.write("\n")

    # Print message
    print("MaxPool2d output saved to maxpool1_pythonresult.txt")

# Define the path to the text file
example_input_path = '/home/rithik/ImageClassification/images/airplane/image_1.txt'
weight_file_path = 'data/features_0_weight.txt'
weight_file_path1 = 'data/features_3_weight.txt'
bias_file_path = 'data/features_0_bias.txt'
bias_file_path1 = 'data/features_3_bias.txt'

# Load and print the shape of the text file
load_and_print_shape(example_input_path)

# Convert the text file to the required shape (1, 3, 32, 32)
loaded_text_data = load_text_data(example_input_path)

# Convert the loaded data to a PyTorch tensor
input_tensor = torch.tensor(loaded_text_data, dtype=torch.float32)

# Load weights and bias
weights, bias = load_weights_bias(weight_file_path, bias_file_path)
weights1, bias1 = load_weights_bias1(weight_file_path1, bias_file_path1)

# Perform convolution with the reshaped input and loaded weights and bias
convolution_with_input(input_tensor, weights, bias, weights1, bias1)

