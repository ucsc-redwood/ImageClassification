#include <iostream>

int main() {
    // Dimensions for the 3D array [batch_size, height, width]
    int batch_size = 2;
    int height = 3;
    int width = 4;

    // Creating a sample 3D array (emulating a tensor)
    int x[2][3][4] = {
        {
            {0, 1, 2, 3},
            {4, 5, 6, 7},
            {8, 9, 10, 11}
        },
        {
            {12, 13, 14, 15},
            {16, 17, 18, 19},
            {20, 21, 22, 23}
        }
    };

    // Flattened size excluding the batch dimension
    int flattened_size = height * width;

    // Creating a 2D array to hold the flattened data
    int y[2][12]; // 2 is the batch size and 12 is the flattened size (height * width)

    // Flattening the array while preserving the batch dimension
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                y[i][j * width + k] = x[i][j][k];
            }
        }
    }

    // Printing the flattened array
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < flattened_size; ++j) {
            std::cout << y[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

