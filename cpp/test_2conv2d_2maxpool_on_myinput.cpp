#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>

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

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
               int pool_size, int stride, float* output_data) {
    // Calculate the dimensions of the output data
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    // Iterate over each channel
    for (int c = 0; c < input_channels; c++) {
        // Iterate over the output height
        for (int h = 0; h < output_height; h++) {
            // Iterate over the output width
            for (int w = 0; w < output_width; w++) {
                float max_val = -FLT_MAX;
                // Iterate over the pooling window
                for (int ph = 0; ph < pool_size; ph++) {
                    for (int pw = 0; pw < pool_size; pw++) {
                        int input_h = h * stride + ph;
                        int input_w = w * stride + pw;
                        if (input_h < input_height && input_w < input_width) {
                            int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
                            max_val = std::max(max_val, input_data[input_index]);
                        }
                    }
                }
                int output_index = c * (output_height * output_width) + h * output_width + w;
                output_data[output_index] = max_val;
            }
        }
    }
}

void conv2d(float* input_data, int image_input_channels, int input_height, int input_width,
            float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
            float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding, 
            bool relu, float* output_data) {
    // Compute output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

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
                                sum += input_data[file_index] * weight_data[weight_index];
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
    int pool_size = 2;  // Max pool window size
    int pool_stride = 2;  // Stride for max pooling
    bool relu = true;

    // Load the image data
    int imageDataSize;
    float* image_data = readDataFromFile("flattened_image_tensor.txt", imageDataSize);

    // Load the weights and data for the first convolutional layer
    int weightDataSize;
    float* weight_data = readDataFromFile("features_0_weight.txt", weightDataSize);
    int biasDataSize;
    float* bias_data = readDataFromFile("features_0_bias.txt", biasDataSize);

    // Load the weights and data for the second convolutional layer
    int secondWeightDataSize;
    float* second_weight_data = readDataFromFile("features_3_weight.txt", secondWeightDataSize);
    int secondBiasDataSize;
    float* second_bias_data = readDataFromFile("features_3_bias.txt", secondBiasDataSize);

    // Load weights and biases for the third convolutional layer
    int thirdWeightDataSize;
    float* third_weight_data = readDataFromFile("features_6_weight.txt", thirdWeightDataSize);
    int thirdBiasDataSize;
    float* third_bias_data = readDataFromFile("features_6_bias.txt", thirdBiasDataSize);

    // Load weights and biases for the fourth convolutional layer
    int fourthWeightDataSize;
    float* fourth_weight_data = readDataFromFile("features_8_weight.txt", fourthWeightDataSize);
    int fourthBiasDataSize;
    float* fourth_bias_data = readDataFromFile("features_8_bias.txt", fourthBiasDataSize);

    // Load weights and biases for the fifth convolutional layer
    int fifthWeightDataSize;
    float* fifth_weight_data = readDataFromFile("features_10_weight.txt", fifthWeightDataSize);
    int fifthBiasDataSize;
    float* fifth_bias_data = readDataFromFile("features_10_bias.txt", fifthBiasDataSize);

    // Allocate memory for convolution output
    int conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    float* conv_output_data = new float[weight_output_channels * conv_output_height * conv_output_width];

    // Call the convolution function
    conv2d(image_data, image_input_channels, input_height, input_width,
           weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
           bias_data, bias_number_of_elements, kernel_size, stride, padding, relu, conv_output_data);

    // Allocate memory for max pooling output
    int pooled_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    int pooled_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    float* maxpool_output_data = new float[weight_output_channels * pooled_output_height * pooled_output_width];

    // Apply max pooling on the convolution output
    maxpool2d(conv_output_data, weight_output_channels, conv_output_height, conv_output_width, pool_size, pool_stride, maxpool_output_data);

    // Allocate memory for the output of the second convolutional layer
    int second_conv_output_height = pooled_output_height;
    int second_conv_output_width = pooled_output_width;
    float* second_conv_output_data = new float[192 * second_conv_output_height * second_conv_output_width];

    // Call the convolution function for the second layer
    conv2d(maxpool_output_data, 64, pooled_output_height, pooled_output_width,
           second_weight_data, 192, 64, 3, 3, 
           second_bias_data, 192, 3, 1, 1,
           true, second_conv_output_data);

    // Allocate memory for the output of the second max pooling layer
    int second_pooled_output_height = (second_conv_output_height - pool_size) / pool_stride + 1;
    int second_pooled_output_width = (second_conv_output_width - pool_size) / pool_stride + 1;
    float* second_maxpool_output_data = new float[192 * second_pooled_output_height * second_pooled_output_width];

    // Apply the second max pooling on the second convolution output
    maxpool2d(second_conv_output_data, 192, second_conv_output_height, second_conv_output_width, pool_size, pool_stride, second_maxpool_output_data);

    // Allocate memory for the output of the third convolutional layer
    int third_conv_output_height = second_pooled_output_height;
    int third_conv_output_width = second_pooled_output_width;
    float* third_conv_output_data = new float[384 * third_conv_output_height * third_conv_output_width];

    // Apply the third convolution
    conv2d(second_maxpool_output_data, 192, second_pooled_output_height, second_pooled_output_width,
           third_weight_data, 384, 192, 3, 3,
           third_bias_data, 384, 3, 1, 1,
           true, third_conv_output_data);


    // Output the result after the third convolutional layer
    std::cout << "Output after the third convolutional layer:" << std::endl;
    for (int c = 0; c < 384; ++c) {  // 384 output channels from the third convolutional layer
        std::cout << "Channel " << c + 1 << ":" << std::endl;
        for (int i = 0; i < third_conv_output_height; ++i) {
            for (int j = 0; j < third_conv_output_width; ++j) {
                std::cout << third_conv_output_data[(c * third_conv_output_height + i) * third_conv_output_width + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    delete[] image_data;
    delete[] weight_data;
    delete[] bias_data;
    delete[] conv_output_data;
    delete[] maxpool_output_data;
    delete[] second_weight_data;
    delete[] second_bias_data;
    delete[] second_conv_output_data;
    delete[] second_maxpool_output_data;

    return 0;
}
