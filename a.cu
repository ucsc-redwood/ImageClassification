#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cfloat>

void readDataFromFile(const std::string& filename, float* data, int dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    int index = 0;
    float value;
    while (index < dataSize && file >> value) {
        data[index++] = value;
    }

    if (index != dataSize) {
        std::cerr << "Error: Data size mismatch. Expected " << dataSize << " elements, but file contains " << index << "." << std::endl;
        return;
    }

    file.close();
}

__global__ void maxpool2d_kernel(float* input_data, int input_channels, int input_height, int input_width,
                                 int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int c = blockIdx.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int w = threadIdx.x + blockIdx.z * blockDim.x;

    if (h < output_height && w < output_width) {
        float max_val = -FLT_MAX;
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int input_h = h * stride + ph;
                int input_w = w * stride + pw;
                if (input_h < input_height && input_w < input_width) {
                    int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
                    max_val = fmaxf(max_val, input_data[input_index]);
                }
            }
        }
        int output_index = c * (output_height * output_width) + h * output_width + w;
        output_data[output_index] = max_val;
    }
}

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
               int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int total_channels = input_channels;
    dim3 threadsPerBlock(16, 16); // Adjust block size as needed
    dim3 numBlocks(total_channels, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (output_width + threadsPerBlock.x - 1) / threadsPerBlock.x);
    maxpool2d_kernel<<<numBlocks, threadsPerBlock>>>(input_data, input_channels, input_height, input_width,
                                                     pool_size, stride, output_data);
}

__global__ void conv2d_kernel(float* input_data, int image_input_channels, int input_height, int input_width,
                              float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
                              float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding,
                              bool relu, float* output_data) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    int out_channel = blockIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.z * blockDim.x;

    if (y < output_height && x < output_width) {
        float sum = 0.0f;
        for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
            for (int ky = 0; ky < weight_height; ++ky) {
                int image_y_base = y * stride + ky - padding;
                for (int kx = 0; kx < weight_width; ++kx) {
                    int image_x = x * stride + kx - padding;
                    if (image_y_base >= 0 && image_y_base < input_height && image_x >= 0 && image_x < input_width) {
                        int input_index = ((in_channel * input_height + image_y_base) * input_width + image_x);
                        int weight_index = ((((out_channel * weight_input_channels) + in_channel) * weight_height + ky) * weight_width + kx);
                        sum += input_data[input_index] * weight_data[weight_index];
                    }
                }
            }
        }
        if (bias_data && out_channel < bias_number_of_elements) {
            sum += bias_data[out_channel];
        }
        if (relu && sum < 0) {
            sum = 0.0f;
        }
        output_data[(out_channel * output_height + y) * output_width + x] = sum;
    }
}

void conv2d(float* input_data, int image_input_channels, int input_height, int input_width,
            float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
            float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding,
            bool relu, float* output_data) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    dim3 threadsPerBlock(16, 16); // Adjust block size as needed
    dim3 numBlocks(weight_output_channels, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (output_width + threadsPerBlock.x - 1) / threadsPerBlock.x);
    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(input_data, image_input_channels, input_height, input_width,
                                                   weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
                                                   bias_data, bias_number_of_elements, kernel_size, stride, padding,
                                                   relu, output_data);

    // Check for kernel launch errors
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }

    // Wait for kernel to finish execution
    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "CUDA error synchronizing threads: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }
}

__global__ void linearLayer_kernel(float* input_data, float* weights, float* bias, float* output_data,
                                   int input_size, int output_size) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index < output_size) {
        float temp = 0;
        for (int j = 0; j < input_size; ++j) {
            temp += input_data[j] * weights[output_index * input_size + j];
        }
        output_data[output_index] = temp + bias[output_index];
    }
}

void linearLayer(float* input_data, float* weights, float* bias, float* output_data,
                 int input_size, int output_size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    linearLayer_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_data, weights, bias, output_data,
                                                           input_size, output_size);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];

    int imageDataSize = 3072;
    float* h_image_data = new float[imageDataSize];
    readDataFromFile(filePath, h_image_data, imageDataSize);

    float* d_image_data;
    cudaMalloc(&d_image_data, imageDataSize * sizeof(float));
    cudaMemcpy(d_image_data, h_image_data, imageDataSize * sizeof(float), cudaMemcpyHostToDevice);

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

    // Load the weights and data for the first convolutional layer
    int weightDataSize = 1728;
    float* h_weight_data = new float[weightDataSize];
    readDataFromFile("data/features_0_weight.txt", h_weight_data, weightDataSize);

    int biasDataSize = 64;
    float* h_bias_data = new float[biasDataSize];
    readDataFromFile("data/features_0_bias.txt", h_bias_data, biasDataSize);

    float* d_weight_data;
    cudaMalloc(&d_weight_data, weightDataSize * sizeof(float));
    cudaMemcpy(d_weight_data, h_weight_data, weightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_bias_data;
    cudaMalloc(&d_bias_data, biasDataSize * sizeof(float));
    cudaMemcpy(d_bias_data, h_bias_data, biasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the second convolutional layer
    int secondWeightDataSize = 110592;
    float* h_second_weight_data = new float[secondWeightDataSize];
    readDataFromFile("data/features_3_weight.txt", h_second_weight_data, secondWeightDataSize);

    int secondBiasDataSize = 192;
    float* h_second_bias_data = new float[secondBiasDataSize];
    readDataFromFile("data/features_3_bias.txt", h_second_bias_data, secondBiasDataSize);

    float* d_second_weight_data;
    cudaMalloc(&d_second_weight_data, secondWeightDataSize * sizeof(float));
    cudaMemcpy(d_second_weight_data, h_second_weight_data, secondWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_second_bias_data;
    cudaMalloc(&d_second_bias_data, secondBiasDataSize * sizeof(float));
    cudaMemcpy(d_second_bias_data, h_second_bias_data, secondBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the third convolutional layer
    int thirdWeightDataSize = 663552;
    float* third_h_weight_data = new float[thirdWeightDataSize];
    readDataFromFile("data/features_6_weight.txt", third_h_weight_data, thirdWeightDataSize);

    int thirdBiasDataSize = 384;
    float* third_h_bias_data = new float[thirdBiasDataSize];
    readDataFromFile("data/features_6_bias.txt", third_h_bias_data, thirdBiasDataSize);

    float* d_third_weight_data;
    cudaMalloc(&d_third_weight_data, thirdWeightDataSize * sizeof(float));
    cudaMemcpy(d_third_weight_data, third_h_weight_data, thirdWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_third_bias_data;
    cudaMalloc(&d_third_bias_data, thirdBiasDataSize * sizeof(float));
    cudaMemcpy(d_third_bias_data, third_h_bias_data, thirdBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the fourth convolutional layer
    int fourthWeightDataSize = 884736;
    float* fourth_h_weight_data = new float[fourthWeightDataSize];
    readDataFromFile("data/features_8_weight.txt", fourth_h_weight_data, fourthWeightDataSize);

    int fourthBiasDataSize = 256;
    float* fourth_h_bias_data = new float[fourthBiasDataSize];
    readDataFromFile("data/features_8_bias.txt", fourth_h_bias_data, fourthBiasDataSize);

    float* d_fourth_weight_data;
    cudaMalloc(&d_fourth_weight_data, fourthWeightDataSize * sizeof(float));
    cudaMemcpy(d_fourth_weight_data, fourth_h_weight_data, fourthWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_fourth_bias_data;
    cudaMalloc(&d_fourth_bias_data, fourthBiasDataSize * sizeof(float));
    cudaMemcpy(d_fourth_bias_data, fourth_h_bias_data, fourthBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the fifth convolutional layer
    int fifthWeightDataSize = 589824;
    float* fifth_h_weight_data = new float[fifthWeightDataSize];
    readDataFromFile("data/features_10_weight.txt", fifth_h_weight_data, fifthWeightDataSize);

    int fifthBiasDataSize = 256;
    float* fifth_h_bias_data = new float[fifthBiasDataSize];
    readDataFromFile("data/features_10_bias.txt", fifth_h_bias_data, fifthBiasDataSize);

    float* d_fifth_weight_data;
    cudaMalloc(&d_fifth_weight_data, fifthWeightDataSize * sizeof(float));
    cudaMemcpy(d_fifth_weight_data, fifth_h_weight_data, fifthWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_fifth_bias_data;
    cudaMalloc(&d_fifth_bias_data, fifthBiasDataSize * sizeof(float));
    cudaMemcpy(d_fifth_bias_data, fifth_h_bias_data, fifthBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and bias for the linear layer
    int linearWeightSize = 40960;
    float* linear_h_weight_data = new float[linearWeightSize];
    readDataFromFile("data/classifier_weight.txt", linear_h_weight_data, linearWeightSize);

    int linearBiasSize = 10;
    float* linear_h_bias_data = new float[linearBiasSize];
    readDataFromFile("data/classifier_bias.txt", linear_h_bias_data, linearBiasSize);

    float* d_linear_weight_data;
    cudaMalloc(&d_linear_weight_data, linearWeightSize * sizeof(float));
    cudaMemcpy(d_linear_weight_data, linear_h_weight_data, linearWeightSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_linear_bias_data;
    cudaMalloc(&d_linear_bias_data, linearBiasSize * sizeof(float));
    cudaMemcpy(d_linear_bias_data, linear_h_bias_data, linearBiasSize * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for convolution output on the device
    int conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    float* d_conv_output_data;
    cudaMalloc(&d_conv_output_data, weight_output_channels * conv_output_height * conv_output_width * sizeof(float));

    auto start_conv1 = std::chrono::steady_clock::now();
    conv2d(d_image_data, image_input_channels, input_height, input_width,
                                       d_weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
                                       d_bias_data, bias_number_of_elements, kernel_size, stride, padding, relu, d_conv_output_data);

    cudaDeviceSynchronize();
    auto end_conv1 = std::chrono::steady_clock::now();

    // Allocate memory for max pooling output on the device
    int pooled_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    int pooled_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    float* d_maxpool_output_data;
    cudaMalloc(&d_maxpool_output_data, weight_output_channels * pooled_output_height * pooled_output_width * sizeof(float));
    
    // Call the max pooling function
    auto start_maxpool1 = std::chrono::steady_clock::now();
    maxpool2d(d_conv_output_data, weight_output_channels, conv_output_height, conv_output_width, pool_size, pool_stride, d_maxpool_output_data);
    auto end_maxpool1 = std::chrono::steady_clock::now();

    // Allocate memory for the output of the second convolutional layer on the device
    int second_conv_output_height = pooled_output_height;
    int second_conv_output_width = pooled_output_width;
    float* d_second_conv_output_data;
    cudaMalloc(&d_second_conv_output_data, 192 * second_conv_output_height * second_conv_output_width * sizeof(float));
    
    // Call the convolution function for the second layer
    auto start_conv2 = std::chrono::steady_clock::now(); 
    conv2d(d_maxpool_output_data, 64, pooled_output_height, pooled_output_width,
                                               d_second_weight_data, 192, 64, 3, 3,
                                               d_second_bias_data, 192, 3, 1, 1,
                                               true, d_second_conv_output_data);
    cudaDeviceSynchronize(); // Ensure all kernel executions are complete before proceeding
    auto end_conv2 = std::chrono::steady_clock::now();

    // Copy the result back to the host
    float* second_conv_output_data = new float[192 * second_conv_output_height * second_conv_output_width];
    cudaMemcpy(second_conv_output_data, d_second_conv_output_data, 192 * second_conv_output_height * second_conv_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate memory for the output of the second max pooling layer on the device
    int second_pooled_output_height = (second_conv_output_height - pool_size) / pool_stride + 1;
    int second_pooled_output_width = (second_conv_output_width - pool_size) / pool_stride + 1;
    float* d_second_maxpool_output_data;
    cudaMalloc(&d_second_maxpool_output_data, 192 * second_pooled_output_height * second_pooled_output_width * sizeof(float));
    
    // Apply the second max pooling on the second convolution output
    auto start_maxpool2 = std::chrono::steady_clock::now(); 
    maxpool2d(d_second_conv_output_data, 192, second_conv_output_height, second_conv_output_width, pool_size, pool_stride, d_second_maxpool_output_data);
    cudaDeviceSynchronize();
    auto end_maxpool2 = std::chrono::steady_clock::now();

    // Copy the result back to the host
    float* second_maxpool_output_data = new float[192 * second_pooled_output_height * second_pooled_output_width];
    cudaMemcpy(second_maxpool_output_data, d_second_maxpool_output_data, 192 * second_pooled_output_height * second_pooled_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image_data);
    cudaFree(d_second_weight_data);
    cudaFree(d_second_weight_data);
    cudaFree(d_third_weight_data);
    cudaFree(d_third_bias_data);
    cudaFree(d_fourth_weight_data);
    cudaFree(d_fourth_bias_data);
    cudaFree(d_linear_weight_data);
    cudaFree(d_linear_bias_data);
    delete[] h_second_weight_data;
    delete[] h_second_bias_data;
    delete[] third_h_weight_data;
    delete[] third_h_bias_data;
    delete[] fourth_h_weight_data;
    delete[] fourth_h_bias_data;
    delete[] linear_h_weight_data;
    delete[] linear_h_bias_data;

    return 0;
}

