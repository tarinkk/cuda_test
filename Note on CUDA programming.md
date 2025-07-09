# Note on CUDA programming

## Table of Contents
- [CUDA Reduction Kernel](#cuda-reduction-kernel)
    - [Baseline](#baseline)

## CUDA Reduction Kernel:

The CUDA kernel performs a parallel reduction operation to compute the sum of elements in an input array. Each block processes a section of the input array, computes the sum of its elements, and writes the result to an output array.

1. **Data Loading**:
    - Each block loads its portion of the global input array (`d_input`) into shared memory (`input_shared`).
    - The starting index for the block's data is calculated as `blockDim.x * blockIdx.x`.
    - Each thread in the block copies one element from global memory to shared memory based on its thread index (threadIdx.x).

    ```cuda
        float *input_begin = d_input + blockDim.x * blockIdx.x;
        __shared__ float input_shared[THREAD_PER_BLOCK];
        input_shared[threadIdx.x] = input_begin[threadIdx.x];
        __syncthreads();
    ```

2. **Parallel Reduction:**

    we will compare several reduction algorithms below.

    - Baseline (Binary-tree type)
    - No divergence branch

3. **Output**:
   - Only the first thread in each block (`threadIdx.x == 0`) writes the block's final sum (stored in `input_shared[0]`) to the output array (`d_output`) at the index corresponding to the block index (`blockIdx.x`).

   ```cuda
   if (threadIdx.x == 0)
       d_output[blockIdx.x] = input_shared[0];
   ```

### Baseline

#### For a block with 8 threads:

1. **Initial State**: Each thread loads one element into shared memory.
   ```
   input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
   ```

2. **Step size 1**: Threads 0, 2, 4, 6 add elements at offset 1 (e.g., thread 0 adds a1 to a0).
   ```
   input_shared: [a0+a1, a1, a2+a3, a3, a4+a5, a5, a6+a7, a7]
   ```

3. **Step size 2**: Threads 0, 4 add elements at offset 2 (e.g., thread 0 adds a2+a3 to a0+a1).
   ```
   input_shared: [a0+a1+a2+a3, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
   ```

4. **Step size 4**: Thread 0 adds the element at offset 4 (a4+a5+a6+a7).
   ```
   input_shared: [a0+a1+a2+a3+a4+a5+a6+a7, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
   ```

#### For a block with 2^N threads
- In each iteration, threads with indices that are multiples of `2*i` (where `i` is the current step size) add the value from the shared memory location at offset `i` to their own.
- After each iteration, `__syncthreads()` ensures all threads complete their additions before proceeding to the next step.
- The step size `i` doubles each iteration (`i *= 2`), reducing the number of active threads until the sum is accumulated in `input_shared[0]`.

    ```cuda
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x % (i * 2) == 0)
        {
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    ```