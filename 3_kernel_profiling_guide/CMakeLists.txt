add_executable(vector_addition.cu)
target_link_libraries(vector_addition PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vector_addition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(vector_addition PRIVATE - lineinfo)


