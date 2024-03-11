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

def load_weights_bias3(weight_file, bias_file):
    # Load weights and bias from text files
    weights = np.loadtxt(weight_file)
    # Reshape the weights to the correct shape
    weights = weights.reshape((384, 192, 3, 3))
    bias = np.loadtxt(bias_file)

    # Print the expected shape of weights and bias
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)

    # Convert to PyTorch tensors
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    return weights_tensor, bias_tensor

def load_weights_bias4(weight_file, bias_file):
    # Load weights and bias from text files
    weights = np.loadtxt(weight_file)
    # Reshape the weights to the correct shape
    weights = weights.reshape((256, 384, 3, 3))
    bias = np.loadtxt(bias_file)

    # Print the expected shape of weights and bias
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)

    # Convert to PyTorch tensors
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    return weights_tensor, bias_tensor

def load_weights_bias5(weight_file, bias_file):
    # Load weights and bias from text files
    weights = np.loadtxt(weight_file)
    # Reshape the weights to the correct shape
    weights = weights.reshape((256, 256, 3, 3))
    bias = np.loadtxt(bias_file)

    # Print the expected shape of weights and bias
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)

    # Convert to PyTorch tensors
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    return weights_tensor, bias_tensor


def load_weights_bias6(weight_file, bias_file):
    # Load weights and bias from text files
    weights = np.loadtxt(weight_file)
    # Reshape the weights to the correct shape
    weights = weights.reshape((10, 12544))
    bias = np.loadtxt(bias_file)

    # Print the expected shape of weights and bias
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)

    # Convert to PyTorch tensors
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    return weights_tensor, bias_tensor

class Conv2dModel(nn.Module):
    def __init__(self, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6):
        super(Conv2dModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=5)
        self.conv1.weight = nn.Parameter(weights1)
        self.conv1.bias = nn.Parameter(bias1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv2.weight = nn.Parameter(weights2)
        self.conv2.bias = nn.Parameter(bias2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv3.weight = nn.Parameter(weights3)
        self.conv3.bias = nn.Parameter(bias3)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4.weight = nn.Parameter(weights4)
        self.conv4.bias = nn.Parameter(bias4)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5.weight = nn.Parameter(weights5)
        self.conv5.bias = nn.Parameter(bias5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(256 * 7 * 7, 10)
        self.linear.weight = nn.Parameter(weights6)
        self.linear.bias = nn.Parameter(bias6)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv1(x)
        print("After Conv1 shape:", x.shape)
        x = self.relu(x)
        x = self.maxpool(x)
        print("After MaxPool1 shape:", x.shape)
        x = self.conv2(x)
        print("After Conv2 shape:", x.shape)
        x = self.relu(x)
        x = self.conv3(x)
        print("After Conv3 shape:", x.shape)
        x = self.relu(x)
        x = self.conv4(x)
        print("After Conv4 shape:", x.shape)
        x = self.relu(x)
        x = self.conv5(x)
        print("After Conv5 shape:", x.shape)
        x = self.relu(x)
        x = self.maxpool2(x)
        print("After MaxPool2 shape dekh idhar:", x.shape)
        x = x.view(x.size(0), -1)
        print("After flattening shape:", x.shape)
        x = self.linear(x)
        print("After Linear layer shape:", x.shape)
        return x

def convolution_with_input(input_tensor, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6):
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    conv_model = Conv2dModel(weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6)
    output = conv_model(input_tensor)
    last_layer_output = output.detach().numpy()
    np.savetxt("output.txt", last_layer_output)
    return last_layer_output

# Define the path to the text file
example_input_path = '/home/rithik/ImageClassification/images/airplane/image_1.txt'
weight_file_path1 = 'data/features_0_weight.txt'
bias_file_path1 = 'data/features_0_bias.txt'
weight_file_path2 = 'data/features_3_weight.txt'
bias_file_path2 = 'data/features_3_bias.txt'
weight_file_path3 = 'data/features_6_weight.txt'
bias_file_path3 = 'data/features_6_bias.txt'
weight_file_path4 = 'data/features_8_weight.txt'
bias_file_path4 = 'data/features_8_bias.txt'
weight_file_path5 = 'data/features_10_weight.txt'
bias_file_path5 = 'data/features_10_bias.txt'
weight_file_path6 = 'data/classifier_weight.txt'
bias_file_path6 = 'data/classifier_bias.txt'

# Load and print the shape of the text file
load_and_print_shape(example_input_path)

# Convert the text file to the required shape (1, 3, 32, 32)
loaded_text_data = load_text_data(example_input_path)

# Load weights and bias
weights1, bias1 = load_weights_bias(weight_file_path1, bias_file_path1)
weights2, bias2 = load_weights_bias1(weight_file_path2, bias_file_path2)
weights3, bias3 = load_weights_bias3(weight_file_path3, bias_file_path3)
weights4, bias4 = load_weights_bias4(weight_file_path4, bias_file_path4)
weights5, bias5 = load_weights_bias5(weight_file_path5, bias_file_path5)
weights6, bias6 = load_weights_bias6(weight_file_path6, bias_file_path6)

# Call the function with appropriate arguments
last_layer_output = convolution_with_input(loaded_text_data, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6)

# Print the output of the last layer
print("Output of the last layer:", last_layer_output)
