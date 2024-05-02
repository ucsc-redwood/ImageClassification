import torch
import torch.nn as nn

# Define the input tensor shape
input_shape = (1, 256, 1, 1)

# Create a random input tensor with the specified shape
input_tensor = torch.randn(*input_shape)

# Define the linear layer with the specified weight and bias shapes
linear_layer = nn.Linear(256, 10)
# Manually set the weight shape to match the specified size
with torch.no_grad():
    linear_layer.weight = nn.Parameter(torch.randn(10, 12544))
    linear_layer.bias = nn.Parameter(torch.randn(10))

# Forward pass the input tensor through the linear layer
output = linear_layer(input_tensor.squeeze())

# Print the output shape
print("Output shape:", output.shape)

