#include <iostream>
#include <fstream>

// Function to read data from a text file into a dynamically allocated array
float* readDataFromFile(const std::string& filename, int& dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return nullptr;
    }

    // Count the number of elements in the file
    float value;
    dataSize = 0;
    while (file >> value) {
        dataSize++;
    }

    // Allocate memory for the data
    float* data = new float[dataSize];

    // Go back to the beginning of the file
    file.clear();
    file.seekg(0, std::ios::beg);

    // Read the data into the array
    int index = 0;
    while (file >> value) {
        data[index++] = value;
    }

    file.close();
    return data;
}

// Function to perform 2D convolution
float* conv2d(float* file_data, int image_input_channels, int input_height, int input_width,
              float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
              float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding, bool relu = false) {
    // Compute output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    // Allocate memory for output
    float* output_data = new float[weight_output_channels * output_height * output_width];

    // Perform convolution
    for (int out_channel = 0; out_channel < weight_output_channels; ++out_channel) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                float sum = 0.0f;
                for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
                    for (int ky = 0; ky < weight_height; ++ky) {
                        for (int kx = 0; kx < weight_width; ++kx) {
                            int image_y = y * stride + ky - padding;
                            int image_x = x * stride + kx - padding;
                            if (image_y >= 0 && image_y < input_height && image_x >= 0 && image_x < input_width) {
                                int file_index = ((in_channel * input_height + image_y) * input_width + image_x);
                                int weight_index = ((((out_channel * weight_input_channels) + in_channel) * weight_height + ky) * weight_width + kx);
                                sum += file_data[file_index] * weight_data[weight_index];
                            }
                        }
                    }
                }
                // Add bias
                if (bias_data && out_channel < bias_number_of_elements)
                    sum += bias_data[out_channel];
                // Apply ReLU if needed
                if (relu && sum < 0)
                    sum = 0.0f;
                // Store result
                output_data[(out_channel * output_height + y) * output_width + x] = sum;
            }
        }
    }

    return output_data;
}

int main() {
    // Initialize parameters
    int image_input_channels = 3;
    int input_height = 32;  // Assuming a 32x32 image based on the provided data length
    int input_width = 32;
    int weight_output_channels = 64;
    int weight_input_channels = 3;
    int weight_height = 3;
    int weight_width = 3;
    int bias_number_of_elements = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    bool relu = true;

    // Load the image data
    int imageDataSize;
    float* image_data = readDataFromFile("flattened_image_tensor.txt", imageDataSize);

    // Load the weight data
    int weightDataSize;
    float* weight_data = readDataFromFile("features_0_weight.txt", weightDataSize);

    // Load the bias data
    int biasDataSize;
    float* bias_data = readDataFromFile("features_0_bias.txt", biasDataSize);

    // Check for data loading errors
    if (imageDataSize != image_input_channels * input_height * input_width ||
        weightDataSize != weight_output_channels * weight_input_channels * weight_height * weight_width ||
        biasDataSize != bias_number_of_elements) {
        std::cerr << "Data size mismatch. Please check the input files and parameters." << std::endl;
        if (image_data) delete[] image_data;
        if (weight_data) delete[] weight_data;
        if (bias_data) delete[] bias_data;
        return -1;
    }

    // Call the convolution function
    float* output_data = conv2d(image_data, image_input_channels, input_height, input_width,
                                weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
                                bias_data, bias_number_of_elements, kernel_size, stride, padding, relu);

    // Output the result
    std::cout << "Output after ReLU activation:" << std::endl;
    for (int c = 0; c < weight_output_channels; ++c) {
        std::cout << "Channel " << c + 1 << ":" << std::endl;
        for (int i = 0; i < ((input_height + 2 * padding - kernel_size) / stride + 1); ++i) {
            for (int j = 0; j < ((input_width + 2 * padding - kernel_size) / stride + 1); ++j) {
                std::cout << output_data[(c * ((input_height + 2 * padding - kernel_size) / stride + 1) + i) * ((input_width + 2 * padding - kernel_size) / stride + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    delete[] image_data;
    delete[] weight_data;
    delete[] bias_data;
    delete[] output_data;

    return 0;
}

