import torch

# Load the model's state dictionary
state_dict = torch.load("cifar_net.pth", map_location=torch.device('cpu'))

# Print information about each layer's parameters
for name, param in state_dict.items():
    print(f"Layer: {name}, Size: {param.size()}, Requires Gradient: {param.requires_grad}")

# Print information about each layer's bias and weight
for name, param in state_dict.items():
    if name.endswith('weight'):
        print(f"Layer: {name}")
        bias_name = name.rsplit('.', 1)[0] + '.bias'
        bias = state_dict.get(bias_name, None)
        print(f"\tWeight Shape: {param.size()}, Bias Shape: {bias.size() if bias is not None else None}")

