import numpy as np

# Load the data from the text file.
# Adjust the delimiter based on how the numbers are separated in your file.
data = np.loadtxt('features_8_weight.txt', delimiter=',')

# Assuming the data is flat and needs to be reshaped to (3, 3, 384, 256).
# This needs to be adjusted if the data structure is different.
reshaped_data = data.reshape((3, 3, 384, 256))

# Print the shape of the reshaped data.
print("Shape of the data:", reshaped_data.shape)

