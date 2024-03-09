#include <iostream>

int main() {
    // Dimensions for the 3D array [batch_size, 1, width]
    int batch_size = 1;
    int height = 1;
    int width = 256;

    // Creating a sample 3D array (emulating a tensor of 1x1x256)
    int x[1][1][256];
    
    // Initializing the array with some values for demonstration
    for (int i = 0; i < width; ++i) {
        x[0][0][i] = i;
    }

    // Creating a 2D array to hold the flattened data
    int y[1][256]; // 1 is the batch size and 256 is the width

    // Flattening the array while preserving the batch dimension
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < width; ++j) {
            y[i][j] = x[i][0][j]; // Directly copying the values as the middle dimension is 1
        }
    }

    // Printing the flattened array
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << y[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

