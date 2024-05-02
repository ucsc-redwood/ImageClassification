import torch
import torch.nn as nn
import numpy as np

def load_and_print_shape(file_path):
    data = np.loadtxt(file_path)
    print("Loaded File Shape:", data.shape)

def load_text_data(file_path):
    data = np.loadtxt(file_path)
    reshaped_data = data.reshape((1, 3, 32, 32))
    print("Reshaped Data shape:", reshaped_data.shape)
    return reshaped_data

def load_weights_bias(weight_file, bias_file, shape):
    weights = np.loadtxt(weight_file).reshape(shape)
    bias = np.loadtxt(bias_file)
    print("Expected Weight Shape:", weights.shape)
    print("Expected Bias Shape:", bias.shape)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)
    return weights_tensor, bias_tensor

class Conv2dModel(nn.Module):
    def __init__(self, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6):
        super(Conv2dModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)  # Adjusted padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.linear = nn.Linear(12544, 10)  # Adjusted for the correct input size

        self.init_weights(weights1, bias1, self.conv1)
        self.init_weights(weights2, bias2, self.conv2)
        self.init_weights(weights3, bias3, self.conv3)
        self.init_weights(weights4, bias4, self.conv4)
        self.init_weights(weights5, bias5, self.conv5)
        self.init_weights(weights6, bias6, self.linear, is_linear=True)

    def init_weights(self, weights, bias, layer, is_linear=False):
        layer.weight = nn.Parameter(weights)
        layer.bias = nn.Parameter(bias)
        if is_linear:
            layer.weight = nn.Parameter(weights.t())  # Transpose weights for linear layer

    def forward(self, x):
        x = self.conv1(x).relu().maxpool2d(kernel_size=3, stride=2)
        x = self.conv2(x).relu().maxpool2d(kernel_size=3, stride=2)
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu().maxpool2d(kernel_size=3, stride=2)
        x = x.view(x.size(0), -1)  # Flatten before passing to the linear layer
        x = self.linear(x)
        return x

# Load the data and the weights/biases
loaded_text_data = load_text_data('/home/rithik/ImageClassification/images/airplane/image_1.txt')
weights1, bias1 = load_weights_bias(weight_file_path1, bias_file_path1, (64, 3, 11, 11))
weights2, bias2 = load_weights_bias(weight_file_path2, bias_file_path2, (192, 64, 5, 5))
weights3, bias3 = load_weights_bias(weight_file_path3, bias_file_path3, (384, 192, 3, 3))
weights4, bias4 = load_weights_bias(weight_file_path4, bias_file_path4, (256, 384, 3, 3))
weights5, bias5 = load_weights_bias(weight_file_path5, bias_file_path5, (256, 256, 3, 3))
weights6, bias6 = load_weights_bias(weight_file_path6, bias_file_path6, (10, 12544))  # Adjusted for the correct shape

# Convert the loaded data to a tensor
input_tensor = torch.tensor(loaded_text_data, dtype=torch.float32)

# Initialize the model and pass the input through it
conv_model = Conv2dModel(weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6)
output = conv_model(input_tensor)
last_layer_output = output.detach().numpy()

print("Output of the last layer:", last_layer_output)

