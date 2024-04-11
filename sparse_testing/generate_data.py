import numpy as np

# Example sizes, adjust as needed
x_size = 256 * 384 * 4  # Adjust based on the actual size needed
s_size = 100  # Number of non-zero elements in sparse matrix
s_indices_size = s_size  # Same as the number of non-zero elements
s_indptr_size = 256 * 384 + 1  # Size of indptr should be rows + 1

# Generate example data
x = np.random.rand(x_size).astype(np.float32)
s = np.random.rand(s_size).astype(np.float32)
s_indices = np.random.randint(0, 9, size=s_indices_size).astype(np.int32)  # Assuming 9 columns based on the C++ code
s_indptr = np.sort(np.random.randint(0, s_size, size=s_indptr_size)).astype(np.int32)
s_indptr[-1] = s_size  # Ensure the last element is the size of s

# Save to files
np.savetxt('x.txt', x, fmt='%f')
np.savetxt('s.txt', s, fmt='%f')
np.savetxt('s_indices.txt', s_indices, fmt='%d')
np.savetxt('s_indptr.txt', s_indptr, fmt='%d')

