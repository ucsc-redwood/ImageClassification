import torch
import numpy as np

# Load the PyTorch model
model_path = 'cifar_net.pth'
model = torch.load(model_path)

# Ensure the model keys are in the expected format (e.g., state_dict)
if isinstance(model, dict) and 'state_dict' in model.keys():
    model = model['state_dict']

# Iterate over each parameter and save it
for name, param in model.items():
    # Convert the tensor to a numpy array
    weight = param.cpu().numpy()
    # Save the numpy array to a text file
    np.savetxt(f'{name.replace(".", "_")}.txt', weight.flatten())

    # Optionally, also save the shape of the weight for later reshaping
    with open(f'{name.replace(".", "_")}_shape.txt', 'w') as f:
        f.write(','.join(map(str, weight.shape)))

