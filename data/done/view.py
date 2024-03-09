import torch

# Creating a sample tensor of shape [2, 3, 4] (for example, [batch_size, channels, features])
x = torch.arange(24).reshape(2, 3, 4)
print("Original Tensor:\n", x)

# Reshape the tensor, flattening all dimensions except for the batch dimension
x_flattened = x.view(x.size(0), -1)
print("Reshaped Tensor:\n", x_flattened)

