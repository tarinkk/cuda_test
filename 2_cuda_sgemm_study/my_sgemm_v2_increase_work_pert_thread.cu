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

template <unsigned int C_TILE_SIZE, unsigned int K_TILE_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm_v2(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    const int tx = threadIdx.x;                 // local column id inside the block 
    const int ty = threadIdx.y;                 // local row id  inside the block
    const int xBase = blockIdx.x * C_TILE_SIZE; // global col of block’s left edge
    const int yBase = blockIdx.y * C_TILE_SIZE; // global row of block’s top edge

    // Pointers to the start of the A and B matrices for this block.
    float *A_ptr_start = A_ptr + yBase * K;
    float *B_ptr_start = B_ptr + xBase;

    // shared memory tiles
    __shared__ float a_shared[C_TILE_SIZE][K_TILE_SIZE];
    __shared__ float b_shared[K_TILE_SIZE][C_TILE_SIZE];

    // running sum for C[y][x]
    float temp[STRIDE][STRIDE] = {0.0};

    // Total number of K tiles (ceil division)
    const int NUM_TILES = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;

    // Loop over K tiles
    for (int tile = 0; tile < NUM_TILES; ++tile)
    {  
        const int kBase = tile * K_TILE_SIZE; // starting K index of this tile
        
        // Load A tile
        for (int kk = tx; kk < K_TILE_SIZE; kk += blockDim.x)
            for (int yy = ty; yy < C_TILE_SIZE; yy += blockDim.y)
            {
                int k = kBase + kk;
                int y = yBase + yy;
                a_shared[yy][kk] = (y < M && k < K) ? A_ptr_start[yy * K + k] : 0.0f;
            }

        // Load B tile
        for (int kk = threadIdx.y; kk < K_TILE_SIZE; kk += blockDim.y) 
            for (int xx = tx; xx < C_TILE_SIZE; xx += blockDim.x)
            {
                int k = kBase + kk;
                int x = xBase + xx;
                b_shared[kk][xx] = (k < K && x < N) ? B_ptr_start[k * N + xx] : 0.0f; 
            }

        __syncthreads();   // ensures tiles fully populated

        // Compute the partial dot product for this tile
        for (int kk = 0; kk < K_TILE_SIZE; ++kk)
            for (int ry = 0; ry < STRIDE; ++ry)
                for (int cx = 0; cx < STRIDE; ++cx)
                {
                    int yy = ty + ry * blockDim.y;
                    int xx = tx + cx * blockDim.x;
                    temp[ry][cx] += a_shared[yy][kk] * b_shared[kk][xx]; 
                }
        __syncthreads(); // avoid data hazard before next load
    }

    // Write back the result in C

    for (int ry = 0; ry < STRIDE; ++ry)
        for (int cx = 0; cx < STRIDE; ++cx)
        {
            int y = yBase + ty + ry * blockDim.y;
            int x = xBase + tx + cx * blockDim.x;
            if (y < M && x < N) C_ptr[y * N + x] = temp[ry][cx];
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
    constexpr int K_TILE_SIZE = 64;
    constexpr int C_TILE_SIZE = 32;
    constexpr int STRIDE = C_TILE_SIZE / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + C_TILE_SIZE - 1) / C_TILE_SIZE, (M + C_TILE_SIZE - 1) / C_TILE_SIZE );
    cuda_sgemm_v2<C_TILE_SIZE, K_TILE_SIZE, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, M, N, K);
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