#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

// CUDA kernel for 2D convolution
__global__ void conv2d_kernel(float* input_data, int image_input_channels, int input_height, int input_width,
                              float* weight_data, int weight_output_channels, int weight_input_channels,
                              int weight_height, int weight_width, float* bias_data, bool relu,
                              float* output_data, int output_height, int output_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height) return;

    for (int out_channel = 0; out_channel < weight_output_channels; ++out_channel) {
        float sum = 0.0f;
        for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
            for (int ky = 0; ky < weight_height; ++ky) {
                for (int kx = 0; kx < weight_width; ++kx) {
                    int image_y = y * stride + ky - weight_height / 2;
                    int image_x = x * stride + kx - weight_width / 2;
                    if (image_y >= 0 && image_y < input_height && image_x >= 0 && image_x < input_width) {
                        int file_index = ((in_channel * input_height + image_y) * input_width + image_x);
                        int weight_index = ((((out_channel * weight_input_channels) + in_channel) * weight_height + ky) * weight_width + kx);
                        sum += input_data[file_index] * weight_data[weight_index];
                    }
                }
            }
        }
        sum += bias_data[out_channel];
        if (relu) sum = max(0.0f, sum);
        output_data[(out_channel * output_height + y) * output_width + x] = sum;
    }
}

// CUDA kernel for 2D max pooling
__global__ void maxpool2d_kernel(float* input_data, int input_channels, int input_height, int input_width,
                                 int pool_size, int stride, float* output_data, int output_height, int output_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height) return;

    for (int c = 0; c < input_channels; c++) {
        float max_val = -FLT_MAX;
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int input_h = y * stride + ph;
                int input_w = x * stride + pw;
                if (input_h < input_height && input_w < input_width) {
                    int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
                    max_val = fmaxf(max_val, input_data[input_index]);
                }
            }
        }
        int output_index = c * (output_height * output_width) + y * output_width + x;
        output_data[output_index] = max_val;
    }
}

// Wrapper function for 2D convolution using CUDA
void conv2d_cuda(float* input_data, int image_input_channels, int input_height, int input_width,
                 float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
                 float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding, 
                 bool relu, float* output_data, int output_height, int output_width) {
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    conv2d_kernel<<<gridDim, blockDim>>>(input_data, image_input_channels, input_height, input_width,
                                          weight_data, weight_output_channels, weight_input_channels,
                                          weight_height, weight_width, bias_data, relu,
                                          output_data, output_height, output_width);
    cudaDeviceSynchronize();
}

// Wrapper function for 2D max pooling using CUDA
void maxpool2d_cuda(float* input_data, int input_channels, int input_height, int input_width,
                    int pool_size, int stride, float* output_data, int output_height, int output_width) {
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    maxpool2d_kernel<<<gridDim, blockDim>>>(input_data, input_channels, input_height, input_width,
                                             pool_size, stride, output_data, output_height, output_width);
    cudaDeviceSynchronize();
}
