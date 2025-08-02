add_executable(my_sgemm_v0_global_memory my_sgemm_v0_global_memory.cu)

target_link_libraries(my_sgemm_v0_global_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v1_shared_memory my_sgemm_v1_shared_memory.cu)

target_link_libraries(my_sgemm_v1_shared_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v1_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v2_increase_work_pert_thread my_sgemm_v2_increase_work_pert_thread.cu)

target_link_libraries(my_sgemm_v2_increase_work_pert_thread PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v2_increase_work_pert_thread PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v3_using_float4 my_sgemm_v3_using_float4.cu)

target_link_libraries(my_sgemm_v3_using_float4 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v3_using_float4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v4_register_outer_product my_sgemm_v4_register_outer_product.cu)

target_link_libraries(my_sgemm_v4_register_outer_product PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v4_register_outer_product PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v5_a_share_transpose my_sgemm_v5_a_share_transpose.cu)

target_link_libraries(my_sgemm_v5_a_share_transpose PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v5_a_share_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v6_double_buffer my_sgemm_v6_double_buffer.cu)

target_link_libraries(my_sgemm_v6_double_buffer PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v6_double_buffer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


