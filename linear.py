import torch
import torch.nn as nn

# Define a fully connected layer with input size of 4 and output size of 3
linear_layer = nn.Linear(4, 3)

# Access the weights and biases of the linear layer
weights = linear_layer.weight
biases = linear_layer.bias

print("Weights:")
print(weights)
print("\nBiases:")
print(biases)

# Create some input data
input_data = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
])

# Apply the linear transformation
output = linear_layer(input_data)

print("\nInput data:")
print(input_data)
print("\nOutput data after linear transformation:")
print(output)

