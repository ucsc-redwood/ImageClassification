#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

void readDataFromFile(const std::string& filename, float* data, int dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (file >> value) {
        if (count < dataSize) {
            data[count++] = value;
        }
    }

    if (count != dataSize) {
        std::cerr << "Data size mismatch. Expected " << dataSize << " elements, but file contains " << count << "." << std::endl;
    }

    file.close();
}

__global__ void maxpool2dKernel(float* input_data, int input_height, int input_width,
                                int pool_size, int stride, float* output_data,
                                int output_height, int output_width, int channels) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (h < output_height && w < output_width && c < channels) {
        float max_val = -FLT_MAX;
        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int ih = h * stride + ph;
                int iw = w * stride + pw;
                if (ih < input_height && iw < input_width) {
                    int idx = (c * input_height + ih) * input_width + iw;
                    max_val = fmaxf(max_val, input_data[idx]);
                }
            }
        }
        output_data[(c * output_height + h) * output_width + w] = max_val;
    }
}

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
               int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    // Allocate memory on the device
    float* d_input_data;
    float* d_output_data;
    cudaMalloc(&d_input_data, input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_output_data, input_channels * output_height * output_width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input_data, input_data, input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x,
                  (output_height + blockSize.y - 1) / blockSize.y,
                  input_channels);

    // Launch the kernel
    maxpool2dKernel<<<gridSize, blockSize>>>(d_input_data, input_height, input_width,
                                             pool_size, stride, d_output_data,
                                             output_height, output_width, input_channels);

    // Copy result back to host
    cudaMemcpy(output_data, d_output_data, input_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_data);
    cudaFree(d_output_data);
}

__global__ void conv2dKernel(float* input_data, int input_channels, int input_height, int input_width,
                             float* weight_data, int output_channels, int kernel_height, int kernel_width,
                             float* bias_data, int stride, int padding,
                             float* output_data, int output_height, int output_width) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < output_width && h < output_height && k < output_channels) {
        float sum = 0.0f;
        for (int c = 0; c < input_channels; ++c) {
            for (int p = 0; p < kernel_height; ++p) {
                for (int q = 0; q < kernel_width; ++q) {
                    int h_offset = h * stride - padding + p;
                    int w_offset = w * stride - padding + q;
                    if (h_offset >= 0 && h_offset < input_height && w_offset >= 0 && w_offset < input_width) {
                        int input_idx = (c * input_height + h_offset) * input_width + w_offset;
                        int weight_idx = ((k * input_channels + c) * kernel_height + p) * kernel_width + q;
                        sum += input_data[input_idx] * weight_data[weight_idx];
                    }
                }
            }
        }
        if (bias_data) {
            sum += bias_data[k];
        }
        output_data[(k * output_height + h) * output_width + w] = sum;
    }
}

void conv2d(float* input_data, int input_channels, int input_height, int input_width,
            float* weight_data, int output_channels, int kernel_height, int kernel_width,
            float* bias_data, int stride, int padding, float* output_data) {
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // Allocate device memory
    float *d_input_data, *d_weight_data, *d_bias_data, *d_output_data;
    cudaMalloc(&d_input_data, input_channels * input_height * input_width * sizeof(float));
    cudaMalloc(&d_weight_data, output_channels * input_channels * kernel_height * kernel_width * sizeof(float));
    cudaMalloc(&d_bias_data, output_channels * sizeof(float));
    cudaMalloc(&d_output_data, output_channels * output_height * output_width * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input_data, input_data, input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_data, weight_data, output_channels * input_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_data, bias_data, output_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x, (output_height + blockSize.y - 1) / blockSize.y, output_channels);

    // Launch the kernel
    conv2dKernel<<<gridSize, blockSize>>>(d_input_data, input_channels, input_height, input_width,
                                          d_weight_data, output_channels, kernel_height, kernel_width,
                                          d_bias_data, stride, padding,
                                          d_output_data, output_height, output_width);

    // Copy the result back to host
    cudaMemcpy(output_data, d_output_data, output_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_data);
    cudaFree(d_weight_data);
    cudaFree(d_bias_data);
    cudaFree(d_output_data);
}

__global__ void linearLayerKernel(float* input_data, float* weights, float* bias, float* output_data, int input_size, int output_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < output_size) {
        float sum = 0;
        for (int j = 0; j < input_size; ++j) {
            sum += input_data[j] * weights[index * input_size + j];
        }
        output_data[index] = sum + bias[index];
    }
}

void linearLayer(float* input_data, float* weights, float* bias, float* output_data, int input_size, int output_size) {
    float *d_input_data, *d_weights, *d_bias, *d_output_data;

    // Allocate memory on the device
    cudaMalloc(&d_input_data, input_size * sizeof(float));
    cudaMalloc(&d_weights, output_size * input_size * sizeof(float));
    cudaMalloc(&d_bias, output_size * sizeof(float));
    cudaMalloc(&d_output_data, output_size * sizeof(float));

    // Copy data to the device
    cudaMemcpy(d_input_data, input_data, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256; // You can choose other block sizes as well
    int numBlocks = (output_size + blockSize - 1) / blockSize;

    // Launch the kernel
    linearLayerKernel<<<numBlocks, blockSize>>>(d_input_data, d_weights, d_bias, d_output_data, input_size, output_size);

    // Copy the result back to the host
    cudaMemcpy(output_data, d_output_data, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_data);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output_data);
}

int main(int argc, char** argv) {
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::steady_clock;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];

    // Initialize parameters
    int image_input_channels = 3;
    int input_height = 32;
    int input_width = 32;
    int weight_output_channels = 64;
    int weight_input_channels = 3;
    int weight_height = 3;
    int weight_width = 3;
    int bias_number_of_elements = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2; 
    int pool_stride = 2;
    bool relu = true;


    // Load the image data
    int imageDataSize = 3072;
    float* image_data = new float[imageDataSize];
    readDataFromFile(filePath, image_data, imageDataSize);
    
    // Allocate GPU memory for image data
    float* d_image_data;
    cudaMalloc(&d_image_data, imageDataSize * sizeof(float));
    // Copy image data from host to device
    cudaMemcpy(d_image_data, image_data, imageDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process for loading weights and biases is similar:
    // Load the weights for the first convolutional layer
    int weightDataSize = 1728;
    float* weight_data = new float[weightDataSize];
    readDataFromFile("data/features_0_weight.txt", weight_data, weightDataSize);
    
    // Allocate GPU memory and copy for weights
    float* d_weight_data;
    cudaMalloc(&d_weight_data, weightDataSize * sizeof(float));
    cudaMemcpy(d_weight_data, weight_data, weightDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load and transfer bias data similarly
    int biasDataSize = 64;
    float* bias_data = new float[biasDataSize];
    readDataFromFile("data/features_0_bias.txt", bias_data, biasDataSize);
    
    float* d_bias_data;
    cudaMalloc(&d_bias_data, biasDataSize * sizeof(float));
    cudaMemcpy(d_bias_data, bias_data, biasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Second convolutional layer
    int secondWeightDataSize = 110592;
    float* second_weight_data = new float[secondWeightDataSize];
    readDataFromFile("data/features_3_weight.txt", second_weight_data, secondWeightDataSize);
    
    float* d_second_weight_data;
    cudaMalloc(&d_second_weight_data, secondWeightDataSize * sizeof(float));
    cudaMemcpy(d_second_weight_data, second_weight_data, secondWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int secondBiasDataSize = 192;
    float* second_bias_data = new float[secondBiasDataSize];
    readDataFromFile("data/features_3_bias.txt", second_bias_data, secondBiasDataSize);
    
    float* d_second_bias_data;
    cudaMalloc(&d_second_bias_data, secondBiasDataSize * sizeof(float));
    cudaMemcpy(d_second_bias_data, second_bias_data, secondBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Third convolutional layer
    int thirdWeightDataSize = 663552;
    float* third_weight_data = new float[thirdWeightDataSize];
    readDataFromFile("data/features_6_weight.txt", third_weight_data, thirdWeightDataSize);
    
    float* d_third_weight_data;
    cudaMalloc(&d_third_weight_data, thirdWeightDataSize * sizeof(float));
    cudaMemcpy(d_third_weight_data, third_weight_data, thirdWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int thirdBiasDataSize = 384;
    float* third_bias_data = new float[thirdBiasDataSize];
    readDataFromFile("data/features_6_bias.txt", third_bias_data, thirdBiasDataSize);
    
    float* d_third_bias_data;
    cudaMalloc(&d_third_bias_data, thirdBiasDataSize * sizeof(float));
    cudaMemcpy(d_third_bias_data, third_bias_data, thirdBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Fourth convolutional layer
    int fourthWeightDataSize = 884736;
    float* fourth_weight_data = new float[fourthWeightDataSize];
    readDataFromFile("data/features_8_weight.txt", fourth_weight_data, fourthWeightDataSize);
    
    float* d_fourth_weight_data;
    cudaMalloc(&d_fourth_weight_data, fourthWeightDataSize * sizeof(float));
    cudaMemcpy(d_fourth_weight_data, fourth_weight_data, fourthWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int fourthBiasDataSize = 256;
    float* fourth_bias_data = new float[fourthBiasDataSize];
    readDataFromFile("data/features_8_bias.txt", fourth_bias_data, fourthBiasDataSize);
    
    float* d_fourth_bias_data;
    cudaMalloc(&d_fourth_bias_data, fourthBiasDataSize * sizeof(float));
    cudaMemcpy(d_fourth_bias_data, fourth_bias_data, fourthBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Fifth convolutional layer
    int fifthWeightDataSize = 589824;
    float* fifth_weight_data = new float[fifthWeightDataSize];
    readDataFromFile("data/features_10_weight.txt", fifth_weight_data, fifthWeightDataSize);
    
    float* d_fifth_weight_data;
    cudaMalloc(&d_fifth_weight_data, fifthWeightDataSize * sizeof(float));
    cudaMemcpy(d_fifth_weight_data, fifth_weight_data, fifthWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int fifthBiasDataSize = 256;
    float* fifth_bias_data = new float[fifthBiasDataSize];
    readDataFromFile("data/features_10_bias.txt", fifth_bias_data, fifthBiasDataSize);
    
    float* d_fifth_bias_data;
    cudaMalloc(&d_fifth_bias_data, fifthBiasDataSize * sizeof(float));
    cudaMemcpy(d_fifth_bias_data, fifth_bias_data, fifthBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Linear layer
    int linearWeightSize = 40960;
    float* linear_weight_data = new float[linearWeightSize];
    readDataFromFile("data/classifier_weight.txt", linear_weight_data, linearWeightSize);
    
    float* d_linear_weight_data;
    cudaMalloc(&d_linear_weight_data, linearWeightSize * sizeof(float));
    cudaMemcpy(d_linear_weight_data, linear_weight_data, linearWeightSize * sizeof(float), cudaMemcpyHostToDevice);
    
    int linearBiasSize = 10;
    float* linear_bias_data = new float[linearBiasSize];
    readDataFromFile("data/classifier_bias.txt", linear_bias_data, linearBiasSize);
    
    float* d_linear_bias_data;
    cudaMalloc(&d_linear_bias_data, linearBiasSize * sizeof(float));
    cudaMemcpy(d_linear_bias_data, linear_bias_data, linearBiasSize * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for convolution output on the device
    float* d_conv_output_data;
    cudaMalloc(&d_conv_output_data, weight_output_channels * conv_output_height * conv_output_width * sizeof(float));
    
    conv2dCUDA(d_image_data, image_input_channels, input_height, input_width,
               d_weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
               d_bias_data, bias_number_of_elements, kernel_size, stride, padding, relu, d_conv_output_data);
    
    // Allocate memory for max pooling output on the device
    float* d_maxpool_output_data;
    cudaMalloc(&d_maxpool_output_data, weight_output_channels * pooled_output_height * pooled_output_width * sizeof(float));
    
    // Call the CUDA max pooling function (assuming it's adapted for CUDA)
    maxpool2dCUDA(d_conv_output_data, weight_output_channels, conv_output_height, conv_output_width, pool_size, pool_stride, d_maxpool_output_data);
    
    // Allocate memory for the output of the second convolutional layer on the device
    float* d_second_conv_output_data;
    cudaMalloc(&d_second_conv_output_data, 192 * second_conv_output_height * second_conv_output_width * sizeof(float));
    
    // Call the second convolution function for CUDA
    conv2dCUDA(d_maxpool_output_data, 64, pooled_output_height, pooled_output_width,
               d_second_weight_data, 192, 64, 3, 3,
               d_second_bias_data, 192, 3, 1, 1,
               true, d_second_conv_output_data);
    
    // Second max pooling layer
    float* d_second_maxpool_output_data;
    cudaMalloc(&d_second_maxpool_output_data, 192 * second_pooled_output_height * second_pooled_output_width * sizeof(float));
    maxpool2dCUDA(d_second_conv_output_data, 192, second_conv_output_height, second_conv_output_width, pool_size, pool_stride, d_second_maxpool_output_data);
    
    // Third convolution layer
    float* d_third_conv_output_data;
    cudaMalloc(&d_third_conv_output_data, 384 * third_conv_output_height * third_conv_output_width * sizeof(float));
    conv2dCUDA(d_second_maxpool_output_data, 192, second_pooled_output_height, second_pooled_output_width,
               d_third_weight_data, 384, 192, 3, 3,
               d_third_bias_data, 384, 3, 1, 1, true, d_third_conv_output_data);
    
    // Fourth convolution layer
    float* d_fourth_conv_output_data;
    cudaMalloc(&d_fourth_conv_output_data, fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width * sizeof(float));
    conv2dCUDA(d_third_conv_output_data, 384, third_conv_output_height, third_conv_output_width,
               d_fourth_weight_data, fourth_conv_output_channels, 384, 3, 3,
               d_fourth_bias_data, fourth_conv_output_channels, 3, 1, 1, true, d_fourth_conv_output_data);
    
    // Fifth convolution layer
    float* d_fifth_conv_output_data;
    cudaMalloc(&d_fifth_conv_output_data, fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width * sizeof(float));
    conv2dCUDA(d_fourth_conv_output_data, 256, fourth_conv_output_height, fourth_conv_output_width,
               d_fifth_weight_data, fifth_conv_output_channels, 256, 3, 3,
               d_fifth_bias_data, fifth_conv_output_channels, 3, 1, 1, true, d_fifth_conv_output_data);
    
    // Max pooling after the fifth convolution layer
    float* d_fifth_maxpool_output_data;
    cudaMalloc(&d_fifth_maxpool_output_data, fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width * sizeof(float));
    maxpool2dCUDA(d_fifth_conv_output_data, fifth_conv_output_channels, fifth_conv_output_height, fifth_conv_output_width, pool_size_after_fifth, pool_stride_after_fifth, d_fifth_maxpool_output_data);


    int totalElements = fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width;
    float* d_flattened_output = d_fifth_maxpool_output_data;  // Directly using the output of the last layer if it's already flattened
    
    // Allocate memory for the output of the linear layer on the device
    int linear_output_size = 10;  // This should match the output size of your linear layer
    float* d_linear_output_data;
    cudaMalloc(&d_linear_output_data, linear_output_size * sizeof(float));
    
    // Apply the linear layer
    linearLayer(d_flattened_output, d_linear_weight_data, d_linear_bias_data, d_linear_output_data, totalElements, linear_output_size);
    
    // Copy the output from the last layer back to the host to determine the predicted class
    float* linear_output_data = new float[linear_output_size];
    cudaMemcpy(linear_output_data, d_linear_output_data, linear_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Determine the predicted class
    int max_index = 0;
    float max_value = linear_output_data[0];
    for (int i = 1; i < linear_output_size; ++i) {
        if (linear_output_data[i] > max_value) {
            max_value = linear_output_data[i];
            max_index = i;
        }
    }
    
    // Output the predicted class
    std::cout << "Predicted Image: ";
    switch (max_index) {
        case 0: std::cout << "airplane"; break;
        case 1: std::cout << "automobile"; break;
        case 2: std::cout << "bird"; break;
        case 3: std::cout << "cat"; break;
        case 4: std::cout << "deer"; break;
        case 5: std::cout << "dog"; break;
        case 6: std::cout << "frog"; break;
        case 7: std::cout << "horse"; break;
        case 8: std::cout << "ship"; break;
        case 9: std::cout << "truck"; break;
        // Add more cases as needed
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_image_data);
    cudaFree(d_weight_data);
    cudaFree(d_bias_data);
    cudaFree(d_second_weight_data);
    cudaFree(d_second_bias_data);
    cudaFree(d_third_weight_data);
    cudaFree(d_third_bias_data);
    cudaFree(d_fourth_weight_data);
    cudaFree(d_fourth_bias_data);
    cudaFree(d_fifth_weight_data);
    cudaFree(d_fifth_bias_data);
    cudaFree(d_linear_weight_data);
    cudaFree(d_linear_bias_data);
    cudaFree(d_linear_output_data);
    
    // Free host memory
    delete[] image_data;
    delete[] weight_data;
    delete[] bias_data;
    delete[] second_weight_data;
    delete[] second_bias_data;
    delete[] third_weight_data;
    delete[] third_bias_data;
    delete[] fourth_weight_data;
    delete[] fourth_bias_data;
    delete[] fifth_weight_data;
    delete[] fifth_bias_data;
    delete[] linear_weight_data;
    delete[] linear_bias_data;
    delete[] linear_output_data;

    return 0;
}
