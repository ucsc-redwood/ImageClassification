#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ifstream weight_file4("features_8_weight.txt");
    float* kernel4 = (float*)malloc(3 * 3 * 384 * 256 * sizeof(float));

    cout << "Kernel defined: " << endl;

    if (!weight_file4.is_open()) {
        cout << "Failed to open file." << endl;
        return 1; // Exit the program if file opening fails
    }

    int count = 0;
    // Read kernel weights from file
    for (int out_channel = 0; out_channel < 256; ++out_channel) {
        for (int in_channel = 0; in_channel < 384; ++in_channel) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    count++;
                    weight_file4 >> kernel4[out_channel * 3 * 3 * 384 + in_channel * 3 * 3 + i * 3 + j];
                }
            }
        }
    }

    cout << "Total weights read: " << count << endl;
    weight_file4.close();

    // Read bias from file
    ifstream bias_file4("features_8_bias.txt");
    float bias4[256];

    if (!bias_file4.is_open()) {
        cout << "Failed to open bias file." << endl;
        free(kernel4); // Release the allocated memory before exiting
        return 1; // Exit the program if file opening fails
    }

    for (int i = 0; i < 256; ++i) {
        bias_file4 >> bias4[i];
    }

    bias_file4.close();

    // Further processing goes here

    // Don't forget to free the allocated memory when you're done with it
    free(kernel4);

    return 0;
}

