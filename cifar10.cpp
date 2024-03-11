#include <iostream>
#include <fstream>
#include <limits>
using namespace std;

// flatten them to 1D
// struct maybe?
// 1D macloocs - GPU
void First_conv2d(const float input[32][32][3], int input_size,
            const float kernel[11][11][3][64],
            const float bias[64], int stride, int padding,
            float output[8][8][64], int &output_size) {
    // Validate the kernel size, stride, and padding
    if (11 > input_size || 11 <= 0 || stride <= 0 || padding < 0) {
        return;
    }

    // Calculate the output size considering the stride and padding
    output_size = (input_size + 2 * padding - 11) / stride + 1;

    for (int out_channel = 0; out_channel < 64; ++out_channel) {
        for (int y_out = 0; y_out < output_size; ++y_out) {
            for (int x_out = 0; x_out < output_size; ++x_out) {
                float sum = 0;
                for (int ky = 0; ky < 11; ++ky) {
                    for (int kx = 0; kx < 11; ++kx) {
                        for (int in_channel = 0; in_channel < 3; ++in_channel) {
                            int input_i = y_out * stride + ky - padding;
                            int input_j = x_out * stride + kx - padding;
                            if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                                sum += input[input_i][input_j][in_channel] * kernel[ky][kx][in_channel][out_channel];
                            }
                        }
                    }
                }
                output[y_out][x_out][out_channel] = sum + bias[out_channel];
            }
        }
    }
    cout << "First_conv2d Input: " << input_size << "x" << input_size << "x3" << " " << "First_conv2d Output: " << output_size << "x" << output_size << "x" << 64 << endl;
    // Print the complete output
    std::cout << "First_conv2d Output:" << std::endl;
    for (int out_channel = 0; out_channel < 64; ++out_channel) {
        std::cout << "Channel " << out_channel << ":" << std::endl;
        for (int y_out = 0; y_out < output_size; ++y_out) {
            for (int x_out = 0; x_out < output_size; ++x_out) {
                std::cout << output[y_out][x_out][out_channel] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void First_ReLU(float output[8][8][64], int output_size) {
    for (int c = 0; c < 64; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                if (output[i][j][c] < 0) {
                    output[i][j][c] = 0; // Replace negative values with zero
                }
            }
        }
    }
    cout << "First_ReLU Input: " << output_size << "x" << output_size << "x" << 64 << " " << "First_ReLU Output: " << output_size << "x" << output_size << "x" << 64 << endl;
    for (int c = 0; c < 64; ++c) {
	    cout << "Channel " << c << ":\n";
	    for (int i = 0; i < output_size; ++i) {
		    for (int j = 0; j < output_size; ++j) {
			    cout << output[i][j][c] << " ";
		    }
		    cout << endl;
	    }
	    cout << endl;
    }
}

void First_maxpool2d(const float input[8][8][64], int input_size,
               int pool_size, int stride, float output[4][4][64], int &output_size) {
    // Correct calculation of output_size
    output_size = (input_size - pool_size) / stride + 1;

    for (int c = 0; c < 64; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                // Initialize max_val to the minimum possible value
                float max_val = std::numeric_limits<float>::lowest();
                for (int m = 0; m < pool_size; ++m) {
                    for (int n = 0; n < pool_size; ++n) {
                        int idx_row = i * stride + m;
                        int idx_col = j * stride + n;
                        // Check if indices are within bounds
                        if (idx_row < input_size && idx_col < input_size) {
                            max_val = std::max(max_val, input[idx_row][idx_col][c]);
                        }
                    }
                }
                output[i][j][c] = max_val;
            }
        }
    }
    cout << "First_maxpool2d Input: " << input_size << "x" << input_size << "x" << 64 << " " << "First_maxpool2d Output: " << output_size << "x" << output_size << "x" << 64 << endl;
    // Print output values
    for (int c = 0; c < 64; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                cout << output[i][j][c] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void Second_conv2d(const float input[4][4][64], int input_size,
            const float kernel[5][5][64][192],
            const float bias[192], int stride, int padding,
            float output[4][4][192], int &output_size) {
    // Validate the kernel size, stride, and padding
/*
    if (5 > input_size || 5 <= 0 || stride <= 0 || padding < 0) {
        std::cout << "Second_conv2d: Invalid kernel size, stride, or padding." << std::endl << endl;
        return;
    }
*/
    // Calculate the output size considering the stride and padding
    output_size = (input_size + 2 * padding - 5) / stride + 1;

    for (int out_channel = 0; out_channel < 192; ++out_channel) {
        for (int y_out = 0; y_out < output_size; ++y_out) {
            for (int x_out = 0; x_out < output_size; ++x_out) {
                float sum = 0;
                for (int ky = 0; ky < 5; ++ky) {
                    for (int kx = 0; kx < 5; ++kx) {
                        for (int in_channel = 0; in_channel < 64; ++in_channel) {
                            int input_i = y_out * stride + ky - padding;
                            int input_j = x_out * stride + kx - padding;
                            if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                                sum += input[input_i][input_j][in_channel] * kernel[ky][kx][in_channel][out_channel];
                            }
                        }
                    }
                }
                output[y_out][x_out][out_channel] = sum + bias[out_channel];
            }
        }
    }
    std::cout << "Second_conv2d Input: " << input_size << "x" << input_size << "x64" << " " << "Second_conv2d Output: " << output_size << "x" << output_size << "x" << 192 << std::endl;
}

void Second_ReLU(float output[4][4][192], int output_size) {
    for (int c = 0; c < 192; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                if (output[i][j][c] < 0) {
                    output[i][j][c] = 0; // Replace negative values with zero
                }
            }
        }
    }
    cout << "Second_ReLU Input: " << output_size << "x" << output_size << "x" << 192 << " " << "Second_ReLU Output: " << output_size << "x" << output_size << "x" << 192 << endl;
}

void Second_maxpool2d(const float input[4][4][192], int input_size,
               int pool_size, int stride, float output[2][2][192], int &output_size) {
    // Correct calculation of output_size
    output_size = (input_size - pool_size) / stride + 1;

    for (int c = 0; c < 192; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                // Initialize max_val to the minimum possible value
                float max_val = std::numeric_limits<float>::lowest();
                for (int m = 0; m < pool_size; ++m) {
                    for (int n = 0; n < pool_size; ++n) {
                        int idx_row = i * stride + m;
                        int idx_col = j * stride + n;
                        // Check if indices are within bounds
                        if (idx_row < input_size && idx_col < input_size) {
                            max_val = std::max(max_val, input[idx_row][idx_col][c]);
                        }
                    }
                }
                output[i][j][c] = max_val;
            }
        }
    }
    cout << "Second_maxpool2d Input: " << input_size << "x" << input_size << "x" << 192 << " " << "Second_maxpool2d Output: " << output_size << "x" << output_size << "x" << 192 << endl;
}

void Third_conv2d(const float input[2][2][192], int input_size,
                  const float kernel[3][3][192][384],
                  const float bias[384], int stride, int padding,
                  float output[2][2][384], int &output_size) {
/*
    // Validate the kernel size, stride, and padding
    if (3 > input_size || 3 <= 0 || stride <= 0 || padding < 0) {
        std::cout << "Invalid kernel size, stride, or padding." << std::endl;
        return;
    }
*/
    // Calculate the output size considering the stride and padding
    output_size = (input_size + 2 * padding - 3) / stride + 1;

    for (int out_channel = 0; out_channel < 384; ++out_channel) {
        for (int y_out = 0; y_out < output_size; ++y_out) {
            for (int x_out = 0; x_out < output_size; ++x_out) {
                float sum = 0;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_channel = 0; in_channel < 192; ++in_channel) {
                            int input_i = y_out * stride + ky - padding;
                            int input_j = x_out * stride + kx - padding;
                            if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                                sum += input[input_i][input_j][in_channel] * kernel[ky][kx][in_channel][out_channel];
                            }
                        }
                    }
                }
                output[y_out][x_out][out_channel] = sum + bias[out_channel];
            }
        }
    }
    std::cout << "Third_conv2d Input: " << input_size << "x" << input_size << "x192" << " " << "Third_conv2d Output: " << output_size << "x" << output_size << "x" << 384 << std::endl;
}

void Third_ReLU(float output[2][2][384], int output_size) {
    for (int c = 0; c < 384; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                if (output[i][j][c] < 0) {
                    output[i][j][c] = 0; // Replace negative values with zero
                }
            }
        }
    }
    cout << "Third_ReLU Input: " << output_size << "x" << output_size << "x" << 384 << " " << "Third_ReLU Output: " << output_size << "x" << output_size << "x" << 384 << endl;
}

void Fourth_conv2d(const float input[2][2][384], int input_size,
                   const float kernel[3][3][384][256],
                   const float bias[256], int stride, int padding,
                   float output[2][2][256], int &output_size) {
    // Validate the kernel size, stride, and padding
/*
    if (3 > input_size || 3 <= 0 || stride <= 0 || padding < 0) {
        std::cout << "Invalid kernel size, stride, or padding." << std::endl;
        return;
    }
*/
    // Calculate the output size considering the stride and padding
    output_size = (input_size + 2 * padding - 3) / stride + 1;

    for (int out_channel = 0; out_channel < 256; ++out_channel) {
        for (int y_out = 0; y_out < output_size; ++y_out) {
            for (int x_out = 0; x_out < output_size; ++x_out) {
                float sum = 0;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_channel = 0; in_channel < 384; ++in_channel) {
                            int input_i = y_out * stride + ky - padding;
                            int input_j = x_out * stride + kx - padding;
                            if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                                sum += input[input_i][input_j][in_channel] * kernel[ky][kx][in_channel][out_channel];
                            }
                        }
                    }
                }
                output[y_out][x_out][out_channel] = sum + bias[out_channel];
            }
        }
    }
    std::cout << "Fourth_conv2d Input: " << input_size << "x" << input_size << "x384" << " " << "Fourth_conv2d Output: " << output_size << "x" << output_size << "x" << 256 << std::endl;
}

void Fourth_ReLU(float output[2][2][256], int output_size) {
    for (int c = 0; c < 256; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                if (output[i][j][c] < 0) {
                    output[i][j][c] = 0; // Replace negative values with zero
                }
            }
        }
    }
    cout << "Fourth_ReLU Input: " << output_size << "x" << output_size << "x" << 256 << " " << "Fourth_ReLU Output: " << output_size << "x" << output_size << "x" << 256 << endl;
}

void Fifth_conv2d(const float input[2][2][256], int input_size,
                   const float kernel[3][3][256][256],
                   const float bias[256], int stride, int padding,
                   float output[2][2][256], int &output_size) {
    // Validate the kernel size, stride, and padding
/*
    if (3 > input_size || 3 <= 0 || stride <= 0 || padding < 0) {
        std::cout << "Invalid kernel size, stride, or padding." << std::endl;
        return;
    }
*/
    // Calculate the output size considering the stride and padding
    output_size = (input_size + 2 * padding - 3) / stride + 1;

    for (int out_channel = 0; out_channel < 256; ++out_channel) {
        for (int y_out = 0; y_out < output_size; ++y_out) {
            for (int x_out = 0; x_out < output_size; ++x_out) {
                float sum = 0;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        for (int in_channel = 0; in_channel < 256; ++in_channel) {
                            int input_i = y_out * stride + ky - padding;
                            int input_j = x_out * stride + kx - padding;
                            if (input_i >= 0 && input_i < input_size && input_j >= 0 && input_j < input_size) {
                                sum += input[input_i][input_j][in_channel] * kernel[ky][kx][in_channel][out_channel];
                            }
                        }
                    }
                }
                output[y_out][x_out][out_channel] = sum + bias[out_channel];
            }
        }
    }
    std::cout << "Fifth_conv2d Input: " << input_size << "x" << input_size << "x256" << " " << "Fifth_conv2d Output: " << output_size << "x" << output_size << "x" << 256 << std::endl;
}

void Fifth_ReLU(float output[2][2][256], int output_size) {
    for (int c = 0; c < 256; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                if (output[i][j][c] < 0) {
                    output[i][j][c] = 0; // Replace negative values with zero
                }
            }
        }
    }
    cout << "Fifth_ReLU Input: " << output_size << "x" << output_size << "x" << 256 << " " << "Fifth_ReLU Output: " << output_size << "x" << output_size << "x" << 256 << endl;
}

void Third_maxpool2d(const float input[2][2][256], int input_size,
               int pool_size, int stride, float output[1][1][256], int &output_size) {
    // Correct calculation of output_size
    output_size = (input_size - pool_size) / stride + 1;

    for (int c = 0; c < 256; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                // Initialize max_val to the minimum possible value
                float max_val = std::numeric_limits<float>::lowest();
                for (int m = 0; m < pool_size; ++m) {
                    for (int n = 0; n < pool_size; ++n) {
                        int idx_row = i * stride + m;
                        int idx_col = j * stride + n;
                        // Check if indices are within bounds
                        if (idx_row < input_size && idx_col < input_size) {
                            max_val = std::max(max_val, input[idx_row][idx_col][c]);
                        }
                    }
                }
                output[i][j][c] = max_val;
            }
        }
    }
    cout << "Third_maxpool2d Input: " << input_size << "x" << input_size << "x" << 256 << " " << "Third_maxpool2d Output: " << output_size << "x" << output_size << "x" << 256 << endl;
}

void view_flatten(const float input[1][1][256], int input_size, float  output[1][256], int output_size) {
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 256; ++j) {
            output[i][j] = input[i][0][j];
        }
    }
    cout << "view_flattend Input: " << input_size << "x" << input_size << "x" << 256 << " " << "view_flattend Output: " << output_size << "x" << 256 << endl;
}

void linear(const float input[1][256], const float kernel[10][256], const float bias[10], float output[10]) {
    for (int i = 0; i < 10; ++i) {
        output[i] = 0;
        for (int j = 0; j < 256; ++j) {
            output[i] += input[0][j] * kernel[i][j];
        }
        output[i] += bias[i];
    }
}

void reshapeInput(float input[3][32 * 32], float reshapedInput[32][32][3]) {
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 32; ++h) {
            for (int w = 0; w < 32; ++w) {
                reshapedInput[h][w][c] = input[c][h * 32 + w];
            }
        }
    }
}


int result(float output[10]) {
    float max = output[0]; // Assume the first element as max initially
    int index = 0;

    // Iterate through the array to find the maximum value
    for (int i = 1; i < 10; i++) {
        if (output[i] > max) {
	    index = i;
            max = output[i];
        }
    }

    cout<<"Max value: " << max << " " << "Image class: " << index << endl;
    return 0;
}

int main() {
    // Read input image from file
    ifstream input_file("/home/rithik/ImageClassification/images/airplane/image_1.txt");
    float input[3][32 * 32]; // Assuming 3 channels and 32x32 pixels
    int input_size = 32; // Size of the image after reshaping

    // Read image data from file
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 32 * 32; ++i) {
            input_file >> input[c][i];
        }
    }
    input_file.close();

    // Reshape input to 3 channels of 32x32 pixels
    float reshapedInput[32][32][3];
    reshapeInput(input, reshapedInput);

//--------------------LAYER ONE CONV INPUTS------------------------//
    // Read kernel weights from file
    ifstream weight_file1("data/features_0_weight.txt");
    float kernel1[11][11][3][64];

    for (int out_channel = 0; out_channel < 64; ++out_channel) {
        for (int in_channel = 0; in_channel < 3; ++in_channel) {
            for (int i = 0; i < 11; ++i) {
                for (int j = 0; j < 11; ++j) {
                    weight_file1 >> kernel1[i][j][in_channel][out_channel];
                }
            }
        }
    }
    weight_file1.close();

    // Read bias from file
    ifstream bias_file1("data/features_0_bias.txt");
    float bias1[64];
    for (int i = 0; i < 64; ++i) {
        bias_file1 >> bias1[i];
    }
    bias_file1.close();

//--------------------LAYER TWO CONV INPUTS-----------------------//
// Read kernel weights from file
    ifstream weight_file2("data/features_3_weight.txt");
    float kernel2[5][5][64][192];

    // Read kernel weights from file
    for (int out_channel = 0; out_channel < 192; ++out_channel) {
        for (int in_channel = 0; in_channel < 64; ++in_channel) {
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    weight_file2 >> kernel2[i][j][in_channel][out_channel];
                }
            }
        }
    }
    weight_file2.close();

    // Read bias from file
    ifstream bias_file2("data/features_3_bias.txt");
    float bias2[192];

    for (int i = 0; i < 192; ++i) {
        bias_file2 >> bias2[i];
    }
    bias_file2.close();

//--------------------LAYER THREE CONV INPUTS---------------------//
    // Read kernel weights from file
    ifstream weight_file3("data/features_6_weight.txt");
    float kernel3[3][3][192][384];

    // Read kernel weights from file
    for (int out_channel = 0; out_channel < 384; ++out_channel) {
        for (int in_channel = 0; in_channel < 192; ++in_channel) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    weight_file3 >> kernel3[i][j][in_channel][out_channel];
                }
            }
        }
    }
    weight_file3.close();

    // Read bias from file
    ifstream bias_file3("data/features_6_bias.txt");
    float bias3[384];

    for (int i = 0; i < 384; ++i) {
        bias_file3 >> bias3[i];
    }
    bias_file3.close();

//--------------------LAYER FOUR CONV INPUTS---------------------//

    ifstream weight_file4("data/features_8_weight.txt");
    float* kernel4 =(float*) malloc(3*3*384*256*sizeof(float));
    
    if (!weight_file4.is_open()) {
    cout << "Failed to open file." << endl;
    }
    
    int count =0;
    // Read kernel weights from file
    for (int out_channel = 0; out_channel < 256; ++out_channel) {
        for (int in_channel = 0; in_channel < 384; ++in_channel) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
			weight_file4 >> kernel4[out_channel]; // Note the order of indices
                }
            }
        }
    }

    weight_file4.close();

    // Read bias from file
    ifstream bias_file4("data/features_8_bias.txt");
    float bias4[256];

    for (int i = 0; i < 256; ++i) {
        bias_file4 >> bias4[i];
    }
    bias_file4.close();

//--------------------LAYER FIFTH CONV INPUTS---------------------//

    ifstream weight_file5("data/features_10_weight.txt");
    float* kernel5 =(float*) malloc(3*3*256*256*sizeof(float));

    // Read kernel weights from file
    for (int out_channel = 0; out_channel < 256; ++out_channel) {
        for (int in_channel = 0; in_channel < 256; ++in_channel) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                        weight_file5 >> kernel5[out_channel]; // Note the order of indices
                }
            }
        }
    }

    weight_file5.close();

    // Read bias from file
    ifstream bias_file5("data/features_8_bias.txt");
    float bias5[256];

    for (int i = 0; i < 256; ++i) {
        bias_file5 >> bias5[i];
    }
    bias_file5.close();

//--------------------LAYER CLASSIFIER INPUTS---------------------//

    // Declare and open the weight file
    std::ifstream weight_file6("data/classifier_weight.txt");
    if (!weight_file6) {
        std::cerr << "Failed to open classifier_weight.txt" << std::endl;
        return 1;  // Return with error code
    }

    // Allocate memory for the large kernel
    float* kernel_large = new float[10 * 12544];

    // Load kernel weights from the file
    for (int i = 0; i < 10 * 12544; ++i) {
        weight_file6 >> kernel_large[i];
    }
    weight_file6.close();  // Close the weight file after reading

    // Define the kernel to be used with the linear function
    float kernel6[10][256];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 256; ++j) {
		kernel6[i][j] = kernel_large[i * 12544 + j];  // Copy the first 256 elements from each row
        }
    }

    delete[] kernel_large;  // Free the memory allocated for kernel_large

    // Open and read the bias values from the classifier_bias file
    std::ifstream bias_file6("data/classifier_bias.txt");
    if (!bias_file6) {
        std::cerr << "Failed to open classifier_bias.txt" << std::endl;
        return 1;  // Return with error code
    }

    float bias6[10];
    for (int i = 0; i < 10; ++i) {
        bias_file6 >> bias6[i];
    }
    bias_file6.close();  // Close the bias file after reading

//--------------------Output Arrays------------------------------//
    // Declare output arrays
    float conv_output1[8][8][64];
    int conv_output_size1;
    // Declare maxpool output arrays
    float maxpool_output1[4][4][64];
    int maxpool_output_size1;

    // Declare output arrays
    float conv_output2[4][4][192];
    int conv_output_size2;
    // Declare maxpool output arrays
    float maxpool_output2[2][2][192];
    int maxpool_output_size2;

    // Declare output arrays
    float conv_output3[2][2][384];
    int conv_output_size3;

    // Declare output arrays
    float conv_output4[2][2][256];
    int conv_output_size4;

    // Declare output arrays
    float conv_output5[2][2][256];
    int conv_output_size5;

    // Declare maxpool output arrays
    float maxpool_output3[1][1][256];
    int maxpool_output_size3;

    // Declare view_flatten output arrays
    float view_flatten_output[1][256];
    int view_flatten_output_size = 1;

    // Declare classifier output arrays
    float classifier_output[10];
    int classifier_output_size;

    // Call conv2d function
    First_conv2d(reshapedInput, 32, kernel1, bias1, 4, 5, conv_output1, conv_output_size1);
    First_ReLU(conv_output1, conv_output_size1);
    First_maxpool2d(conv_output1, conv_output_size1, 2, 2, maxpool_output1, maxpool_output_size1);
    Second_conv2d(maxpool_output1, 4, kernel2, bias2, 1, 2, conv_output2, conv_output_size2);
    Second_ReLU(conv_output2, conv_output_size2);
    Second_maxpool2d(conv_output2, conv_output_size2, 2, 2, maxpool_output2, maxpool_output_size2);
    Third_conv2d(maxpool_output2, 2, kernel3, bias3, 1, 1, conv_output3, conv_output_size3);
    Third_ReLU(conv_output3, conv_output_size3);
    Fourth_conv2d(conv_output3, 2, (const float(*)[3][384][256])kernel4, bias4, 1, 1, conv_output4, conv_output_size4);
    Fourth_ReLU(conv_output4, conv_output_size4);
    Fifth_conv2d(conv_output4, 2, (const float(*)[3][256][256])kernel5, bias5, 1, 1, conv_output5, conv_output_size5);
    Fifth_ReLU(conv_output5, conv_output_size5);
    Third_maxpool2d(conv_output5, conv_output_size5, 2, 2, maxpool_output3, maxpool_output_size3);
    view_flatten(maxpool_output3, maxpool_output_size3, view_flatten_output, view_flatten_output_size);
    linear(view_flatten_output, kernel6, bias6, classifier_output);
    result(classifier_output);
}
