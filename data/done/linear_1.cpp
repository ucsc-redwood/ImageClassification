#include <iostream>
#include <fstream>

void linear(const float input[1][256], const float kernel[10][256], const float bias[10], float output[10]) {
    for (int i = 0; i < 10; ++i) {
        output[i] = 0;
        for (int j = 0; j < 256; ++j) {
            output[i] += input[0][j] * kernel[i][j];
        }
        output[i] += bias[i];
    }
}

int main() {
    // Example input array
    float input[1][256];
    for(int i = 0; i < 256; ++i) {
        input[0][i] = 1.0f; // Fill with some example data
    }

    // Allocate memory for the large kernel
    float* kernel_large = new float[10 * 12544];

    // Load kernel weights
    std::ifstream weight_file6("classifier_weight.txt");
    if (!weight_file6) {
        std::cerr << "Failed to open classifier_weight.txt" << std::endl;
        delete[] kernel_large;
        return 1;
    }

    for (int i = 0; i < 10 * 12544; ++i) {
        weight_file6 >> kernel_large[i];
    }
    weight_file6.close();

    // Define the kernel to be used with the linear function
    float kernel[10][256];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 256; ++j) {
            kernel[i][j] = kernel_large[i * 12544 + j];  // Take the first 256 elements from each row
        }
    }

    // Free the large kernel memory
    delete[] kernel_large;

    // Load bias values
    float bias[10];
    std::ifstream bias_file6("classifier_bias.txt");
    if (!bias_file6) {
        std::cerr << "Failed to open classifier_bias.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < 10; ++i) {
        bias_file6 >> bias[i];
    }
    bias_file6.close();

    // Output array
    float output[10];

    // Apply the linear function
    linear(input, kernel, bias, output);

    // Print the output
    for (int i = 0; i < 10; ++i) {
        std::cout << "Output[" << i << "]: " << output[i] << std::endl;
    }

    return 0;
}

