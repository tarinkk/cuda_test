#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define THREAD_PER_BLOCK 128

// Each block will compute the sum (reduce) of its section of the input array.
// The result from each block is written to the output array.
__global__ void reduce_v4(float *d_input, float *d_output)
{
    // Find the start of this block's data in the global array
    // subarray_size = 2 * blockDim.x
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    // Initialize shared memory for this block
    __shared__ float input_shared[THREAD_PER_BLOCK];
    // Each thread sum two elements from global memory into shared memory
    input_shared[threadIdx.x] = input_begin[threadIdx.x] + input_begin[threadIdx.x + blockDim.x];
    __syncthreads(); // Ensure all threads have written their data to shared memory

    // if (threadIdx.x == 0, 1)
    //     input_shared[threadIdx.x] += input_shared[threadIdx.x + 2];
    // if (threadIdx.x == 0)
    //     input_shared[threadIdx.x] += input_shared[threadIdx.x + 1];
    // half the step size for each iteration
    // On each step i, threads with indices less than i add the value 
    // at index threadIdx.x to the value at threadIdx.x + i
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {   
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads(); // Synchronize to make sure all threads are done before the next step
    }
    // Only the first thread in each block writes the final sum to the output array
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = input_shared[0];
}

// Compare two arrays for near-equality; returns true if all elements are nearly the same
bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(out[i] - res[i]) > 0.005)
            return false;
    }
    return true;
}

int main()
{
    // Number of input elements to reduce
    const int N = 32 * 1024 * 1024;

    // --- Memory allocation on CPU ---
    float *input = (float *)malloc(N * sizeof(float));

    // --- Memory allocation on GPU ---
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    // Number of blocks: each block reduces THREAD_PER_BLOCK elements
    int block_num = N / (2 * THREAD_PER_BLOCK);

    // Size of each subarray that each block will process
    int subarray_size = N / block_num;

    // Allocate memory for block-wise reduction results on CPU
    float *output = (float *)malloc(block_num * sizeof(float));

    // Allocate memory for block-wise reduction results on GPU
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    // Allocate memory for CPU-computed reference result
    float *result = (float *)malloc(block_num * sizeof(float));

    // Initialize input array with random values between -1 and 1
    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // --- CPU computation of per-block reductions (reference answer) ---
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < subarray_size; j++)
        {
            cur += input[i * subarray_size + j];
        }
        result[i] = cur;
    }

    // Copy input data from CPU to GPU
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure CUDA kernel launch: one block per chunk, THREAD_PER_BLOCK threads per block
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    // Launch reduction kernel
    reduce_v4<<<Grid, Block>>>(d_input, d_output);

    // Copy the per-block sums from GPU back to CPU
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare GPU and CPU results
    if (check(output, result, block_num))
        printf("The answer is correct\n");
    else
    {
        printf("The answer is wrong\n");
        for (int i = 0; i < block_num; i++)
        {
            printf("%lf ", output[i]);
        }
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free CPU memory
    free(input);
    free(output);
    free(result);

    return 0;
}