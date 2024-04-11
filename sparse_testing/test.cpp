#include <iostream>

// Function signature for sparse_conv_if
int sparse_conv_if(float* x, float* s, int* s_indices, int* s_indptr, float* out);

int main() {
    // Input Matrix (3x3)
    float input[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // CSR Representation for a dense 3x3 filter
    int s_indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // Column indices of non-zero elements
    int s_indptr[] = {0, 9}; // Indptr array indicating the start of each row
    float s[] = {1, 1, 1, 1, 1, 1, 1, 1, 1}; // Values corresponding to non-zero elements

    // Output Feature Map (1D array)
    float output[1] = {0};

    // Perform Convolution
    sparse_conv_if(&input[0][0], s, s_indices, s_indptr, output);

    // Display Output Feature Map
    std::cout << "Output Feature Map:\n";
    for (int i = 0; i < 1; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

int sparse_conv_if(
    float*  x,
    float*  s,
    int*    s_indices,
    int*    s_indptr,
    float*  out)
{
    // Since it's a 3x3 input, we'll have a single output value.
    int outputSize = 1;
    for(int k = 0; k < outputSize; k++){
        // Since it's a single output, these will be 0
        int j = 0;
        int i = 0;

        for (int kk = s_indptr[k]; kk < s_indptr[k + 1]; kk++){
            int col = s_indices[kk];
            float val = s[kk];
            int w = col / 3;
            int z = col % 3;

            // Since we're only dealing with a single output, the convolution is direct
            out[k] += val * x[w * 3 + z];
        }
    }
    return 0;
}

