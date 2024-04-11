import numpy as np
from scipy.signal import convolve2d

# Input Matrix (3x3)
input_matrix = np.array([[1, 2, 3], 
                         [4, 5, 6], 
                         [7, 8, 9]])

# Define the kernel (3x3 with all ones)
# This is equivalent to the s array in the C++ code with all ones
kernel = np.array([[1, 1, 1], 
                   [1, 1, 1], 
                   [1, 1, 1]])

# Perform convolution
# 'valid' mode means no padding is applied and the output size is reduced
output = convolve2d(input_matrix, kernel, mode='valid')

# Display Output Feature Map
print("Output Feature Map:")
print(output)

