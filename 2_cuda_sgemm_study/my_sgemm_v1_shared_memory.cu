#include <cstdio>               // printf
#include <cuda.h>
#include <stdlib.h>             // malloc, free, drand48, abs
#include <cuda_runtime.h>       // cudaMalloc, cudaMemcpy

// Utility: Fill a host matrix with random numbers in the range [-1, 1]
void random_matrix(int M, int N, float *A_ptr)
{
    for (int m = 0; m < M; m++)
        for(int n = 0; n < N; n++)
        {
            A_ptr[m * N + n] = 2.0 * (float)drand48() - 1.0;
        } 
}

// Reference SGEMM on CPU: triple-nested loop (O(M.N.K))
void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                C_ptr[m * N + n] += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
        }
}

// Compare two matrices element-wise; return max difference
float compare_matrices(int M, int N, float *C_gpu, float *C_cpu)
{
    float diff = 0.0f;
    float max_diff = 0.0f;
    for (int m = 0; m < M; m++)
        for(int n = 0; n < N; n++)
        {
            diff = abs(C_gpu[m * N + n] - C_cpu[m * N + n]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    return max_diff;
}
template <unsigned int BLOCK_SIZE, unsigned int K_TILE_SIZE>
__global__ void cuda_sgemm_v1(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // Thread <-> output element mapping
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= M) return; //boundary check

    // Pointers to the start of the A and B matrices for this block.
    float *A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K;
    float *B_ptr_start = B_ptr + blockDim.x * blockIdx.x;

    // shared memory tiles
    __shared__ float a_shared[BLOCK_SIZE][K_TILE_SIZE];
    __shared__ float b_shared[K_TILE_SIZE][BLOCK_SIZE];

    // running sum for C[y][x]
    float temp = 0.f;

    // Total number of K tiles (ceil division)
    const int NUM_TILES = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;

    // Loop over K tiles
    for (int tile = 0; tile < NUM_TILES; ++tile)
    {  
        const int kBase = tile * K_TILE_SIZE; // starting K index of this tile
        
        // Load of A tile
        // Each thread loads elements (y, kBase + kk)
        // kk is the local index within the tile
        for (int kk = threadIdx.x; kk < K_TILE_SIZE; kk += BLOCK_SIZE)
        {
            int k = kBase + kk;
            a_shared[threadIdx.y][kk] = (y < M && k < K) ? A_ptr_start[threadIdx.y * K + k] : 0.0f;
        }

        // Load of B tile
        for (int kk = threadIdx.y; kk < K_TILE_SIZE; kk += BLOCK_SIZE)
        {
            int k = kBase + kk;
            b_shared[kk][threadIdx.x] = (k < K && x < N) ? B_ptr_start[k * N + threadIdx.x] : 0.0f;
        }

        __syncthreads();   // ensures tiles fully populated

        // Compute the partial dot product for this tile
        for (int k = 0; k < K_TILE_SIZE; k++)
        {
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads(); // avoid data hazard before next load
    }
    
    if (y < M && x < N)
    {
        // Store the result in C
        C_ptr[y * N + x] += temp;
    }
}

int main()
{
    // Matrix sizes
    int M = 512;
    int N = 512;
    int K = 512;

    // Host memory allocation
    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);
    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);
    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    // Generate random inputs
    random_matrix(M, K, matrix_A_host);
    random_matrix(K, N, matrix_B_host);
    printf("Random matrices generated.\n");

    // Initialize output matrices to zero
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C);
    printf("Memory allocated and initialized.\n");

    // Device memory allocation
    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    // Copy inputs to GPU
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);
    printf("Data copied to device.\n");

    // CPU (reference) computation
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, M, N, K);
    printf("CPU SGEMM completed.\n");
    
    // Launch GPU SGEMM kernel
    constexpr int BLOCK_SIZE = 16;
    // for simplicity, we assume K are multiple of K_TILE_SIZE and K_TILE_SIZE is a multiple of BLOCK_SIZE
    constexpr int K_TILE_SIZE = 32;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cuda_sgemm_v1<BLOCK_SIZE, K_TILE_SIZE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, M, N, K);
    printf("GPU SGEMM kernel launched.\n");

    // Retrieve result from GPU
    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    printf("Data copied from device to host.\n");

    // Verify results
    float diff = compare_matrices(M, N, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    printf("Comparison of GPU and CPU results completed.\n");

    if (diff > 0.5f)
    {
        printf("Error: GPU and CPU results do not match! Difference: %f\n", diff);

    }
    else
    {
        printf("Success: GPU and CPU results match! Difference: %f\n", diff);
    }

    // Cleanup
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    printf("Memory freed.\n");
    return 0;
}