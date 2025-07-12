# Note on CUDA programming

## Table of Contents
1. [CUDA Reduction Kernel](#cuda-reduction-kernel)
    - [Baseline](#baseline)
    - [No Divergence Branch](#no-divergence-branch)
    - [No Bank Conflict](#no-bank-conflict)
    - [Add During Load](#add-during-load)
    - [Unroll Last Warp](#unroll-last-warp)
    - [Shuffle](#shuffle)

2. [SGEMM](#sgemm)
## CUDA Reduction Kernel

The CUDA kernel performs a parallel reduction operation to compute the sum of elements in an input array. Each block processes a section of the input array, computes the sum of its elements, and writes the result to an output array.

**Input**: an array of length N.

**Config**: M: split the array into M portions, or the number of blocks. (N is divisible by M.)


**Output**: an array of length M.

1. **Data Loading (baseline)**:

    ```cuda
    // Find the start of this block's data in the global array
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    // Initialize shared memory for this block
    __shared__ float input_shared[THREAD_PER_BLOCK];
    // Threadwisely load data from global memory to shared memory
    input_shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads(); // Ensure all threads have written their data to shared memory
    ```

2. **Parallel Reduction:**

    we will compare several optimization strategies below.

    - Baseline
    - No divergence branch
    - No bank conflict
    - Add during Load
    - Unroll last warp
    - Shuffle

3. **Output**:

    ```cuda
    // Only the first thread in each block writes the final sum to the output array
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = input_shared[0];
    ```

### Baseline
1. **Algorithm**

    `D = blockDim = THREAD_PER_BlOCK= N/M`
    
    - **Step size 1**: `threadIdx.x` = 0, 2, 4, 6, ..., D-2,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 1];
        __syncthreads();
        ```
    - **Step size 2**: `threadIdx.x` = 0, 4, 8, 12,..., D-4,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 2];
        __syncthreads();
        ```
    - **Step size k**: `threadIdx.x` = 0, 2k, 4k, 6k,..., D-2k,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + k];
        __syncthreads();
        ```
    - **Step size D/2**: `threadIdx.x` = 0,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/2];
        __syncthreads();
        ```

    
1. **Reduce a subarray of length 8 (Visualization)**:

    D = 8:

    - **Initial State**: Each thread loads one element into shared memory.
        ```
        input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```

    - **Step size 1**: Threads 0, 2, 4, 6 add elements at offset 1 (e.g., thread 0 adds a1 to a0).
        ```
        input_shared: [a0+a1, a1, a2+a3, a3, a4+a5, a5, a6+a7, a7]
        ```

    - **Step size 2**: Threads 0, 4 add elements at offset 2 (e.g., thread 0 adds a2+a3 to a0+a1).
        ```
        input_shared: [a0+a1+a2+a3, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

    - **Step size 4**: Thread 0 adds the element at offset 4 (a4+a5+a6+a7).
        ```
        input_shared: [a0+a1+a2+a3+a4+a5+a6+a7, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

2. **Key Code**:

    ```cuda
    // double the step size i for each iteration
    for (int i = 1; i < blockDim.x; i *= 2)
    {   
        //threads with indices that are multiples of 2*i (where 
        //i is the current step size) add the value from the 
        //shared memory location at offset i to their own.
        if (threadIdx.x % (i * 2) == 0)
        {
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    ```


### No Divergence Branch
Assign work to threads based on a stride that aligns with warp boundaries, ensuring all threads in a warp perform similar operations.
1. **Algorithm**

    `D = blockDim = THREAD_PER_BlOCK= N/M`
    
    - **Step size 1**: `threadIdx.x` = 0, 1, 2, 3, ..., D/2 - 1,

        ```
        input_shared[2 * threadIdx.x] += input_shared[2 * threadIdx.x + 1];
        __syncthreads();
        ```
    - **Step size 2**: `threadIdx.x` = 0, 1, 2, 3,..., D/4 - 1,

        ```
        input_shared[4 * threadIdx.x] += input_shared[4 * threadIdx.x + 2];
        __syncthreads();
        ```
    - **Step size k**: `threadIdx.x` = 0, 1, 2, 3,..., D/(2*k) - 1,

        ```
        input_shared[2*k * threadIdx.x] += input_shared[2*k *threadIdx.x + k];
        __syncthreads();
        ```
    - **Step size D/2**: `threadIdx.x` = 0,

        ```
        input_shared[D * threadIdx.x] += input_shared[D * threadIdx.x + D/2];
        __syncthreads();
        ```

    
1. **Reduce a subarray of length 8 (Visualization)**:
    
    D = 8:

    - **Initial state**: Each thread loads one element into shared memory.
        ```
        input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```

    - **Step size 1**: Threads 0, 1, 2, 3 add elements at indices `threadIdx.x * 2 and threadIdx.x * 2 + 1` (e.g., thread 0 adds a1 to a0, thread 1 adds a3 to a2).
        ```
        input_shared: [a0+a1, a1, a2+a3, a3, a4+a5, a5, a6+a7, a7]
        ```

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x * 4` and `threadIdx.x * 4 + 2` (e.g., thread 0 adds a2+a3 to a0+a1).
        ```
        input_shared: [a0+a1+a2+a3, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

    - **Step size 4**: Thread 0 adds elements at indices `threadIdx.x * 8` and `threadIdx.x * 8 + 4` (i.e., a4+a5+a6+a7 to a0+a1+a2+a3).
        ```
        input_shared: [a0+a1+a2+a3+a4+a5+a6+a7, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

2. **Key Code**:

    ```cuda
    // double the step size i for each iteration
    // On each step, threads with indices less than blockDim.x/(2*i) add the value 
    // at index threadIdx.x*2*i to the value at threadIdx.x*2*i + i
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x <  blockDim.x / (i * 2))
        {   
            int index = threadIdx.x * 2 * i;
            input_shared[index] += input_shared[index + i];
        }
        __syncthreads(); // Synchronize to make sure all threads are done before the next step
    }
    ```
3. **Discussions**:
    - **Thread Utilization**:
    Comparing to the baseline, for each iteration, only the first `blockDim.x/(2*i)` threads are active, utilizing a contiguous block of threads. This reduces thread divergence within warps, as active threads are grouped together, improving warp execution efficiency.
    - **Bank Conflicts**:
    Let us assume `THREAD_PER_BLOCK` is large:
        1. `i = 1`:
            - **Active Threads**: `threadIdx.x < 64 / (2 * 1) = 32` (threads 0 to 31). 
            -  `index = threadIdx.x * 2 * 1 = 2 * threadIdx.x`
            - Thread 0: Accesses `input_shared[0]` (bank 0) and `input_shared[1]` (bank 1).
            - Thread 16: Accesses `input_shared[32]` (bank 0) and `input_shared[33]` (bank 1).
            - Thread N and Thread N+16 have conflict.
        2. `i = 2`:
            - **Active Threads**: `threadIdx.x < 64 / (2 * 2) = 16` (threads 0 to 31). 
            -  `index = threadIdx.x * 2 * 2 = 4 * threadIdx.x`
            - Threads N, N+8, N+16, N+32 have conflicts.
        3. `i = 2^k`, `i <= 16`: 
            - Threads `{N+ n (16/i) | n = 0,1, ...(2i-1)}`  have conflicts.
        4.  `i>=16`:
            - All threads have bank conflicts between each other. 

### No Bank Conflict
To avoid bank conflict, it's better to ensure threads in a warp access consecutive addresses that map to different banks.
1. **Algorithm**

    `D = blockDim = THREAD_PER_BlOCK= N/M`
    
    - **Step size D/2**: `threadIdx.x` = 0, 1, 2, 3, ..., D/2 - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/2];
        __syncthreads();
        ```
    - **Step size D/4**: `threadIdx.x` = 0, 1, 2, 3,..., D/4 - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/4];
        __syncthreads();
        ```
    - **Step size k**: `threadIdx.x` = 0, 1, 2, 3,..., k - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + k];
        __syncthreads();
        ```
    - **Step size 1**: `threadIdx.x` = 0,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 1];
        __syncthreads();
        ```
1. **Reduce a subarray of length 8 (Visualization)**:
    
    With warp size 2, addresses 0, 2, 4, 6 are in bank 0, addresses 1,3,5,7 are in bank 1.

    - **Initial state**: Each thread loads one element into shared memory.
        ```
        input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```

    - **Step size 4**: Threads 0, 1, 2, 3 add elements at indices `threadIdx.x and threadIdx.x + 4` (e.g., thread 0 adds a4 to a0, thread 1 adds a5 to a1).
        ```
        input_shared: [a0+a4, a1+a5, a2+a6, a3+a7, a4, a5, a6, a7]
        ```

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x` and `threadIdx.x + 2` (e.g., thread 0 adds a0+a4 to a2+a6).
        ```
        input_shared: [a0+a4+a2+a6, a1+a5+a3+a7, a2+a6, a3+a7, a4, a5, a6, a7]
        ```

    - **Step size 1**: Thread 0 adds elements at indices `threadIdx.x ` and `threadIdx.x + 1` (i.e., a0+a4+a2+a6 to a1+a5+a3+a7).
        ```
        input_shared: [a0+a4+a2+a6+a1+a5+a3+a7, a1+a5+a3+a7, a2+a6, a3+a7, a4, a5, a6, a7]
        ```
2. **Key Code**:
    ```cuda
    // half the step size for each iteration
    // for each step size i, threads with indices less than i add the value 
    // at index threadIdx.x to the value at threadIdx.x + i
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {   
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads(); // Synchronize to make sure all threads are done before the next step
    }
    ```
3. **Discussion**:
    - When step size i >= 32, thread `threadIdx.x` access bank `threadIdx.x % 32`, ensuring threads within the same warp access distinct banks.
    - When step size i < 32, data is stored in addresses mapped to different banks, preventing bank conflicts.

### Add During Load
Each thread sums L elements from global memory before storing the result in shared memory

1. **Algorithm**
    Given fixed N and N, D is given by.

    `D = blockDim = THREAD_PER_BlOCK= N/(L*M)`

    - **Add During Load**: `threadIdx.x` = 0, 1, 2, 3, ..., D-1

        ```
        input_shared[threadIdx.x] += input_begin[threadIdx.x + 0];
        input_shared[threadIdx.x] += input_begin[threadIdx.x + D];
        input_shared[threadIdx.x] += input_begin[threadIdx.x + 2*D];
        input_shared[threadIdx.x] += input_begin[threadIdx.x + 3*D];
        ...
        input_shared[threadIdx.x] += input_begin[threadIdx.x + (L-1)*D];
        __syncthreads();
        ```
    
    - **Step size D/2**: `threadIdx.x` = 0, 1, 2, 3, ..., D/2 - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/2];
        __syncthreads();
        ```
    - **Step size D/4**: `threadIdx.x` = 0, 1, 2, 3,..., D/4 - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/4];
        __syncthreads();
        ```
    - **Step size k**: `threadIdx.x` = 0, 1, 2, 3,..., k - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + k];
        __syncthreads();
        ```
    - **Step size 1**: `threadIdx.x` = 0,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 1];
        __syncthreads();
        ```

1. **Reduce a subarray of length 8 (Visualization)**:

    L = 2, `THREAD_PER_BlOCK` is halved. 

    - **Initial state**: Starting from an array in global memory:
        ```
        input_begin: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```
    
    - **Add during load**: Threads 0, 1, 2, 3 sum elements at indices `threadIdx.x and threadIdx.x + 4` in global memory and then store the result at index `threadIdx.x in the shared memory.
        ```
        input_shared: [a0+a4, a1+a5, a2+a6, a3+a7]
        ```

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x` and `threadIdx.x + 2` (e.g., thread 0 adds a0+a4 to a2+a6).
        ```
        input_shared: [a0+a4+a2+a6, a1+a5+a3+a7, a2+a6, a3+a7]
        ```

    - **Step size 1**: Thread 0 adds elements at indices `threadIdx.x ` and `threadIdx.x + 1` (i.e., a0+a4+a2+a6 to a1+a5+a3+a7).
        ```
        input_shared: [a0+a4+a2+a6+a1+a5+a3+a7, a1+a5+a3+a7, a2+a6, a3+a7]
        ```

2. **Key code**
    
    L = 2:
    ```cuda
    // Find the start of this block's data in the global array
    // subarray_size = 2 * blockDim.x
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    // Initialize shared memory for this block
    __shared__ float input_shared[THREAD_PER_BLOCK];
    // Each thread sum two elements from global memory into shared memory
    input_shared[threadIdx.x] = input_begin[threadIdx.x] + input_begin[threadIdx.x + blockDim.x];
    __syncthreads(); // Ensure all threads have written their data to shared memory
    ```

3. **Discussion**
    - This strategy reduces shared memory accesses and synchronization, while leaves global memory access unchanged, decreasing the total computation time.
    - It reduce shared memory usage as well as threads per blocks, potentially allowing more blocks to run concurrently on a streaming multiprocessor.

### Unroll Last Warp
This strategy stops the generic loop when only a single warp (32 threads) remains and finishes the last six steps with hand-unrolled, warp-synchronous additions that need no further barriers.

1. **Algorithm**
    Given fixed N M and L, D is given by.

    `D = blockDim = THREAD_PER_BlOCK= N/(L*M)`

    - **Add During Load**: `threadIdx.x` = 0, 1, 2, 3, ..., D-1

        ```
        input_shared[threadIdx.x] += input_begin[threadIdx.x + 0];
        input_shared[threadIdx.x] += input_begin[threadIdx.x + D];
        input_shared[threadIdx.x] += input_begin[threadIdx.x + 2*D];
        input_shared[threadIdx.x] += input_begin[threadIdx.x + 3*D];
        ...
        input_shared[threadIdx.x] += input_begin[threadIdx.x + (L-1)*D];
        __syncthreads();
        ```
    
    - **Step size D/2**: `threadIdx.x` = 0, 1, 2, 3, ..., D/2 - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/2];
        __syncthreads();
        ```
    - **Step size D/4**: `threadIdx.x` = 0, 1, 2, 3,..., D/4 - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + D/4];
        __syncthreads();
        ```
    - **Step size k>32**: `threadIdx.x` = 0, 1, 2, 3,..., k - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + k];
        __syncthreads();
        ```
    - **Step size k<=32**: `threadIdx.x` = 0, 1, 2, 3,..., k - 1,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + k];
        ```
    - **Step size 1**: `threadIdx.x` = 0,

        ```
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 1];
        ```

1. **Reduce a subarray of length 8 (Visualization)**:

    Supposing warp size is 2, 

    - **Initial state**: Starting from an array in global memory:
        ```
        input_begin: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```
    
    - **Add during load**: Threads 0, 1, 2, 3 sum elements at indices `threadIdx.x and threadIdx.x + 4` in global memory and then store the result at index `threadIdx.x in the shared memory. 
        ```
        input_shared: [a0+a4, a1+a5, a2+a6, a3+a7]
        ```
        sync

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x` and `threadIdx.x + 2` (e.g., thread 0 adds a0+a4 to a2+a6). 
        ```
        input_shared: [a0+a4+a2+a6, a1+a5+a3+a7, a2+a6, a3+a7]
        ```
        *no need to sync*

    - **Step size 1**: Thread 0, 1 adds elements at indices `threadIdx.x ` and `threadIdx.x + 1` (i.e., a0+a4+a2+a6 to a1+a5+a3+a7).
        ```
        input_shared: [a0+a4+a2+a6+a1+a5+a3+a7, a1+a5+a3+a7+a2+a6, a2+a6, a3+a7]
        ```
        *no need to sync, threads in a warp do the same operation*

2. **Key code**
    ```cuda
    for (int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if (threadIdx.x < i)
        {   
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads(); // Synchronize to make sure all threads are done before the next step
    }
    // Perform final reduction steps for the first 32 threads using unrolled loop
    // This avoids synchronization overhead for the last few steps
    if (threadIdx.x < 32)
    {
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 32];
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 16];
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 8];
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 4];
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 2];
        input_shared[threadIdx.x] += input_shared[threadIdx.x + 1];
    }
    ```


3. **Improvement (128-thread block)**

    | Metric                                       | `reduce_v4`                              | `reduce_v5`                              |
    | -------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
    | Barrier calls / block                        | 8                                       | 2                                        |
    | Generic-loop iterations                      | 7                                       | 1                                        |
    | Adds executed by each *participating* thread | identical (every element is summed once) |                                          |
    | Compiler-generated branches in warp phase    | 6 (loop-back + if-tests)                 | 1 (`if(threadIdx.x<32)`)                 |
    | Visible memory operations in warp phase      | 6 loads + 6 stores guarded by the loop   | 6 loads + 6 stores, but no loop overhead |

    - Cuts the number of full-block synchronizations.
    - Removes the loop-control overhead once only one warp is active.


### Shuffle
We can use the shuffle function to achieve the same result. 

1. **Algorithm**
    Given fixed N M and L=2, D is given by.

    `D = blockDim = THREAD_PER_BlOCK= N/(2*M)`

    Choose M,N, such that D <= 1024

    - **Add During Load**: `threadIdx.x` = 0, 1, 2, 3, ..., D-1

        ```
        sum += input_begin[threadIdx.x + 0];
        sum += input_begin[threadIdx.x + D];
        ```
    
    - **Shuffle in Warp**: 
        All threads are working

        ```
        sum += __shfl_down_sync(0xffffffff, sum, 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1)
        ```
    - **Save in shared memory**: 

        Initialize the shared memory

        ```
        __shared__ float warpLevelSums[32];
        ```
        Define warpId and laneId

        ```
        const int warpId = threadIdx.x // 32;
        const int laneId = threadIdx.x % 32;
        ```
        Save reduced sum in the shared memory
        `laneId == 0`
        ```
        warpLevelSums[warpId] = (warpId < D/32) ? sum : 0.f;    
        __syncthreads();
        ```
    - **Shuffle in Warp again**: When `warpId == 0`

        ```
        sum = warpLevelSums[laneId]
        sum += __shfl_down_sync(0xffffffff, sum, 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1)
        ```




## SGEMM

## Reference:
1. [[CUDA]Reduce规约求和（已完结~）](https://www.bilibili.com/video/BV1HvBSY2EJW?spm_id_from=333.788.videopod.episodes&vd_source=aa41d00aebd84e6f99f529df7f83258a)
2. [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
3. [[CUDA编程]束内洗牌函数 (Warp Shuffle Functions)](https://zhuanlan.zhihu.com/p/669957986)
4. [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788)