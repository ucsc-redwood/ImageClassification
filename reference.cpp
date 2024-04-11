#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include "omp.h"

#define debug std::cout<<"here?\n";

#define BATCHSIZE 1
#define NNZ 5668
#define DENSE_SIZE 1536
#define OUT_SIZE 1024

#define NUM_SAMPLES 1.0

int num_comps[] = {1,2,1,2,4,2,1,2,1};
int ccs[][16] = {{1,1,0,0}, {1,0,0,0, 1,1,0,1 }, {1,0,0,1},{0,1,0,0, 1,1,1,0},
                {0,0,0,0, 0,1,0,1, 1,0,1,0, 1,1,1,1}, {0,0,0,1, 1,0,1,1},
                {0,1,1,0},{0,0,1,0, 0,1,1,1},{0,0,1,1}};

int lba[] = {0, 13578, 23020, 37711, 57997, 67145, 72213, 89524, 256*384};


void read_n_floats(int n, float *data, std::string filename){
        std::ifstream infile(filename, std::ios::in);
        int i=0; 
        while(infile.good() && i < n){
                infile >> data[i];
                i++;
        }
        infile.close();
}

void read_n_ints(int n, int *data, std::string filename){
        std::ifstream infile(filename, std::ios::in);
        int i=0; 
        while(infile.good() && i < n){
                infile >> data[i];
                i++;
        }
        infile.close();
}

int sparse_conv(
        float*  x, 
        float*  s,
        int*    s_indices, 
        int*    s_indptr,
        float*  out)
{
        //int stepnum = 0; // for debugging

        for(int k=0; k<256*384; k++){
                int j = k/384;
                int i = k%384;
                //std::cout << k << " " << j << " " << i << '\n';
                for (int kk = s_indptr[k]; kk < s_indptr[k+1]; kk++){
                        int col = s_indices[kk];
                        float val = s[kk];
                        int w   = col/3;
                        int z   = col%3;

                        for(int c = 0; c < 4*num_comps[col]; c += 4){
                                int m = ccs[col][c];
                                int n = ccs[col][c+1];
                                int o = ccs[col][c+2];
                                int p = ccs[col][c+3];
                                out[j*2*2 + m*2 + n] += val * x[i*2*2 + o*2 + p];

                                //std::cout << "step: " << stepnum << "\n";
                                //std::cout << j << " " <<  i << " " << col << " " << w << " " << z << " " << m << " " << n << " " << o << " " << p << " " << val << "\n";
                                //stepnum++;
                        }
                }
        }
        return 0;
}

int sparse_conv_if(
        float*  x, 
        float*  s,
        int*    s_indices, 
        int*    s_indptr,
        float*  out)
{
        for(int k=0; k<256*384; k++){
                int j = k/384;
                int i = k%384;

                for (int kk = s_indptr[k]; kk < s_indptr[k+1]; kk++){
                        int col = s_indices[kk];
                        float val = s[kk];
                        int w   = col/3;
                        int z   = col%3;

                        /*for(int c = 0; c < 4*num_comps[col]; c += 4){
                                int m = ccs[col][c];
                                int n = ccs[col][c+1];
                                int o = ccs[col][c+2];
                                int p = ccs[col][c+3];
                                out[j*2*2 + m*2 + n] += val * x[i*2*2 + o*2 + p];

                                //std::cout << "step: " << stepnum << "\n";
                                //std::cout << j << " " <<  i << " " << col << " " << w << " " << z << " " << m << " " << n << " " << o << " " << p << " " << val << "\n";
                                //stepnum++;
                        }*/
                        /*int num_comps[] = {1,2,1,2,4,2,1,2,1};
                        int ccs[][16] = {{1,1,0,0}, {1,0,0,0, 1,1,0,1 }, {1,0,0,1},{0,1,0,0, 1,1,1,0},
                {0,0,0,0, 0,1,0,1, 1,0,1,0, 1,1,1,1}, {0,0,0,1, 1,0,1,1},
                {0,1,1,0},{0,0,1,0, 0,1,1,1},{0,0,1,1}};
        */
                        if(col == 0){
                                //{1,1,0,0}
                                out[j*2*2 + 1*2 + 1] += val * x[i*2*2 + 0*2 + 0];
                        }else if(col == 1){
                                //{1,0,0,0, 1,1,0,1 }
                                out[j*2*2 + 1*2 + 0] += val * x[i*2*2 + 0*2 + 0];
                                out[j*2*2 + 1*2 + 1] += val * x[i*2*2 + 0*2 + 1];
                        }else if(col == 2){
                                //{1,0,0,1}
                                out[j*2*2 + 1*2 + 0] += val * x[i*2*2 + 0*2 + 1];
                        }else if(col == 3){
                                //{0,1,0,0, 1,1,1,0}
                                out[j*2*2 + 0*2 + 1] += val * x[i*2*2 + 0*2 + 0];
                                out[j*2*2 + 1*2 + 1] += val * x[i*2*2 + 1*2 + 0];
                        }else if(col == 4){
                                //{0,0,0,0, 0,1,0,1, 1,0,1,0, 1,1,1,1}
                                out[j*2*2 + 0*2 + 0] += val * x[i*2*2 + 0*2 + 0];
                                out[j*2*2 + 0*2 + 1] += val * x[i*2*2 + 0*2 + 1];
                                out[j*2*2 + 1*2 + 0] += val * x[i*2*2 + 1*2 + 0];
                                out[j*2*2 + 1*2 + 1] += val * x[i*2*2 + 1*2 + 1];
                        }else if(col == 5){
                                // {0,0,0,1, 1,0,1,1}
                                out[j*2*2 + 0*2 + 0] += val * x[i*2*2 + 0*2 + 1];
                                out[j*2*2 + 1*2 + 0] += val * x[i*2*2 + 1*2 + 1];
                        }else if(col == 6){
                                //{0,1,1,0}
                                out[j*2*2 + 0*2 + 1] += val * x[i*2*2 + 1*2 + 0];
                        }else if(col == 7){
                                //{0,0,1,0, 0,1,1,1}
                                out[j*2*2 + 0*2 + 0] += val * x[i*2*2 + 1*2 + 0];
                                out[j*2*2 + 0*2 + 1] += val * x[i*2*2 + 1*2 + 1];
                        }else if(col == 8){
                                //{0,0,1,1}}
                                out[j*2*2 + 0*2 + 0] += val * x[i*2*2 + 1*2 + 1];
                        }
                }
        }
        return 0;
}

int sparse_conv_parallel(
        float*  x, 
        float*  s,
        int*    s_indices, 
        int*    s_indptr,
        float*  out)
{
        //int stepnum = 0; // for debugging

        #pragma omp parallel for
        for(int k=0; k<256*384; k++){
                int j = k/384;
                int i = k%384;
                //std::cout << k << " " << j << " " << i << '\n';
                for (int kk = s_indptr[k]; kk < s_indptr[k+1]; kk++){
                        int col = s_indices[kk];
                        float val = s[kk];
                        int w   = col/3;
                        int z   = col%3;

                        for(int c = 0; c < 4*num_comps[col]; c += 4){
                                int m = ccs[col][c];
                                int n = ccs[col][c+1];
                                int o = ccs[col][c+2];
                                int p = ccs[col][c+3];
                                out[j*2*2 + m*2 + n] += val * x[i*2*2 + o*2 + p];

                                //std::cout << "step: " << stepnum << "\n";
                                //std::cout << j << " " <<  i << " " << col << " " << w << " " << z << " " << m << " " << n << " " << o << " " << p << " " << val << "\n";
                                //stepnum++;
                        }
                }
        }
        return 0;
}

int sparse_conv_batched(
        float*  x, 
        float*  s,
        int*    s_indices, 
        int*    s_indptr,
        float*  out)
{
        //int stepnum = 0; // for debugging

        //#pragma omp parallel for
        for(int k=0; k<256*384; k++){
                for(int b=0; b<BATCHSIZE; b++){
                        int j = k/384;
                        int i = k%384;
                        //std::cout << k << " " << j << " " << i << '\n';
                        for (int kk = s_indptr[k]; kk < s_indptr[k+1]; kk++){
                                int col = s_indices[kk];
                                float val = s[kk];
                                int w   = col/3;
                                int z   = col%3;

                                for(int c = 0; c < 4*num_comps[col]; c += 4){
                                        int m = ccs[col][c];
                                        int n = ccs[col][c+1];
                                        int o = ccs[col][c+2];
                                        int p = ccs[col][c+3];
                                        out[b*256*2*2 + j*2*2 + m*2 + n] += val * x[b*384*2*2 + i*2*2 + o*2 + p];

                                        //std::cout << "step: " << stepnum << "\n";
                                        //std::cout << j << " " <<  i << " " << col << " " << w << " " << z << " " << m << " " << n << " " << o << " " << p << " " << val << "\n";
                                        //stepnum++;
                                }
                        }
                }
        }
        return 0;
}

int sparse_conv_batched_parallel(
        float*  x, 
        float*  s,
        int*    s_indices, 
        int*    s_indptr,
        float*  out)
{
        //int stepnum = 0; // for debugging

        #pragma omp parallel for
        for(int k=0; k<256*384; k++){
                for(int b=0; b<BATCHSIZE; b++){
                        int j = k/384;
                        int i = k%384;
                        //std::cout << k << " " << j << " " << i << '\n';
                        for (int kk = s_indptr[k]; kk < s_indptr[k+1]; kk++){
                                int col = s_indices[kk];
                                float val = s[kk];
                                int w   = col/3;
                                int z   = col%3;

                                for(int c = 0; c < 4*num_comps[col]; c += 4){
                                        int m = ccs[col][c];
                                        int n = ccs[col][c+1];
                                        int o = ccs[col][c+2];
                                        int p = ccs[col][c+3];
                                        out[b*256*2*2 + j*2*2 + m*2 + n] += val * x[b*384*2*2 + i*2*2 + o*2 + p];

                                        //std::cout << "step: " << stepnum << "\n";
                                        //std::cout << j << " " <<  i << " " << col << " " << w << " " << z << " " << m << " " << n << " " << o << " " << p << " " << val << "\n";
                                        //stepnum++;
                                }
                        }
                }
        }
        return 0;
}

int sparse_conv_batched_parallel_lb(
        float*  x, 
        float*  s,
        int*    s_indices, 
        int*    s_indptr,
        float*  out)
{
        //int stepnum = 0; // for debugging

        #pragma omp parallel for
        for(int t=0; t<8; t++){
                for(int k=lba[t]; k<lba[t+1]; k++){
                        for(int b=0; b<BATCHSIZE; b++){
                                int j = k/384;
                                int i = k%384;
                                //std::cout << k << " " << j << " " << i << '\n';
                                for (int kk = s_indptr[k]; kk < s_indptr[k+1]; kk++){
                                        int col = s_indices[kk];
                                        float val = s[kk];
                                        int w   = col/3;
                                        int z   = col%3;

                                        for(int c = 0; c < 4*num_comps[col]; c += 4){
                                                int m = ccs[col][c];
                                                int n = ccs[col][c+1];
                                                int o = ccs[col][c+2];
                                                int p = ccs[col][c+3];
                                                out[b*256*2*2 + j*2*2 + m*2 + n] += val * x[b*384*2*2 + i*2*2 + o*2 + p];
                                        }
                                }
                        }
                }
        }
        return 0;
}

int main(int argc, char** argv){
        float *d   = new float[BATCHSIZE * DENSE_SIZE];
        float *s   = new float[NNZ];
        float *out = new float[BATCHSIZE * OUT_SIZE];
        float *ref = new float[BATCHSIZE * OUT_SIZE];
        int   s_indices[NNZ];
        int   s_indptr[(384*256) + 1];

        //read data for input and output reference
        read_n_floats(BATCHSIZE * DENSE_SIZE, d,  "batched_input_ref.txt");
        read_n_floats(BATCHSIZE * OUT_SIZE, ref,  "batched_output_ref.txt");

        //read csr data
        read_n_floats(NNZ, s, "csr_data.txt");
        read_n_ints(NNZ, s_indices, "csr_indices.txt");
        read_n_ints((384*256)+1, s_indptr, "csr_indptr.txt");

        //memset
        for(int i=0; i<OUT_SIZE; i++) out[i] = 0;

        using std::chrono::duration_cast;
        using std::chrono::nanoseconds;
        typedef std::chrono::high_resolution_clock clock;
        auto start = clock::now();

        for(int samp=0; samp<(int)NUM_SAMPLES; samp++)
                sparse_conv(d, s, s_indices, s_indptr, out);

        auto end = clock::now();
        std::cout << (duration_cast<nanoseconds>(end-start).count() / 1000000.0) / NUM_SAMPLES << " ms per kernel with batch size " << BATCHSIZE << "\n";

        int errcount = 0;
        for(int i=0; i< OUT_SIZE; i++){
                //std::cout << ref[i] << " "  << out[i] << "\n";
                if (std::abs(out[i]-ref[i])>0.1){
                        std::cout << "res != ref at " << i << "\n";
                        errcount++;
                }
        }
        if(errcount == 0) std::cout << "no errors!\n";
        else std::cout << "ERROR COUNT: " << errcount << "\n";  
}
