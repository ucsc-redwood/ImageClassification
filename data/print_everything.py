import torch

# Load the .pth file
model_path = 'cifar_net.pth'
checkpoint = torch.load(model_path)

# Print keys to see what's stored in the checkpoint
print("Keys in the checkpoint:", checkpoint.keys())

# Access and print each component
for key in checkpoint.keys():
    print("\nComponent:", key)
    if 'state_dict' in key:
        # If it's a state_dict, print its keys and shapes
        state_dict = checkpoint[key]
        for param_tensor in state_dict:
            print(param_tensor, "\t", state_dict[param_tensor].shape)
    else:
        # For other components, simply print their values
        print(checkpoint[key])

