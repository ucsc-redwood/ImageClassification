import torch
import torch.nn as nn

# Number of input features and classes
num_features = 12544
num_classes = 10

# Define the classifier layer
classifier_layer = nn.Linear(num_features, num_classes)

# Initialize weights and biases with custom values
# For example, using normal distribution for weights and zeros for biases
classifier_layer.weight.data = torch.randn(num_classes, num_features)  # Random normal initialization
classifier_layer.bias.data = torch.zeros(num_classes)  # Zeros initialization

# Assuming x is the input tensor to the classifier layer
# This line is illustrative; replace it with actual computation in your model
# x = some_flattening_or_pooling_operation(previous_layer_output)

# Compute the output
# output = classifier_layer(x)

# Print shapes to verify
print("Weight shape:", classifier_layer.weight.shape)
print("Bias shape:", classifier_layer.bias.shape)

