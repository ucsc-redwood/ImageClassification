#include <iostream>
#include <fstream>
#include <string>

void read_from_file(const std::string& filename, float*& out, int& size) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return;
    }

    file >> size; // First line contains size
    out = new float[size];

    for (int i = 0; i < size; ++i) {
        file >> out[i];
    }
}

void read_from_file(const std::string& filename, int*& out, int& size) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return;
    }

    file >> size; // First line contains size
    out = new int[size];

    for (int i = 0; i < size; ++i) {
        file >> out[i];
    }
}

int sparse_conv_if() {
    float *x = nullptr, *s = nullptr, *out = nullptr;
    int *s_indices = nullptr, *s_indptr = nullptr;
    int x_size, s_size, s_indices_size, s_indptr_size, out_size;

    read_from_file("x.txt", x, x_size);
    read_from_file("s.txt", s, s_size);
    read_from_file("s_indices.txt", s_indices, s_indices_size);
    read_from_file("s_indptr.txt", s_indptr, s_indptr_size);

    out_size = 256 * 384 * 4; // Adjust based on the actual size needed
    out = new float[out_size]();

    // Sparse convolution logic
    for (int k = 0; k < 256 * 384; k++) {
        int j = k / 384;
        int i = k % 384;

        for (int kk = s_indptr[k]; kk < s_indptr[k + 1]; kk++) {
            int col = s_indices[kk];
            float val = s[kk];
            switch (col) {
            case 0:
                // Corresponds to {1,1,0,0}
                out[j * 2 * 2 + 1 * 2 + 1] += val * x[i * 2 * 2 + 0 * 2 + 0];
                break;
            case 1:
                // Corresponds to {1,0,0,0, 1,1,0,1 }
                out[j * 2 * 2 + 1 * 2 + 0] += val * x[i * 2 * 2 + 0 * 2 + 0];
                out[j * 2 * 2 + 1 * 2 + 1] += val * x[i * 2 * 2 + 0 * 2 + 1];
                break;
            case 2:
                // Corresponds to {1,0,0,1}
                out[j * 2 * 2 + 1 * 2 + 0] += val * x[i * 2 * 2 + 0 * 2 + 1];
                break;
            case 3:
                // Corresponds to {0,1,0,0, 1,1,1,0}
                out[j * 2 * 2 + 0 * 2 + 1] += val * x[i * 2 * 2 + 0 * 2 + 0];
                out[j * 2 * 2 + 1 * 2 + 1] += val * x[i * 2 * 2 + 1 * 2 + 0];
                break;
            case 4:
                // Corresponds to {0,0,0,0, 0,1,0,1, 1,0,1,0, 1,1,1,1}
                out[j * 2 * 2 + 0 * 2 + 0] += val * x[i * 2 * 2 + 0 * 2 + 0];
                out[j * 2 * 2 + 0 * 2 + 1] += val * x[i * 2 * 2 + 0 * 2 + 1];
                out[j * 2 * 2 + 1 * 2 + 0] += val * x[i * 2 * 2 + 1 * 2 + 0];
                out[j * 2 * 2 + 1 * 2 + 1] += val * x[i * 2 * 2 + 1 * 2 + 1];
                break;
            case 5:
                // Corresponds to {0,0,0,1, 1,0,1,1}
                out[j * 2 * 2 + 0 * 2 + 0] += val * x[i * 2 * 2 + 0 * 2 + 1];
                out[j * 2 * 2 + 1 * 2 + 0] += val * x[i * 2 * 2 + 1 * 2 + 1];
                break;
            case 6:
                // Corresponds to {0,1,1,0}
                out[j * 2 * 2 + 0 * 2 + 1] += val * x[i * 2 * 2 + 1 * 2 + 0];
                break;
            case 7:
                // Corresponds to {0,0,1,0, 0,1,1,1}
                out[j * 2 * 2 + 0 * 2 + 0] += val * x[i * 2 * 2 + 1 * 2 + 0];
                out[j * 2 * 2 + 0 * 2 + 1] += val * x[i * 2 * 2 + 1 * 2 + 1];
                break;
            case 8:
                // Corresponds to {0,0,1,1}
                out[j * 2 * 2 + 0 * 2 + 0] += val * x[i * 2 * 2 + 1 * 2 + 1];
                break;
        	}
	 }
    }

    // Cleanup
    delete[] x;
    delete[] s;
    delete[] s_indices;
    delete[] s_indptr;
    delete[] out;

    return 0;
}

int main() {
    sparse_conv_if();
    return 0;
}

