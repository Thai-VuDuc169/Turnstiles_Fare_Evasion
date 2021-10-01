cmake_minimum_required(VERSION 2.8)
project(test_infer2features)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(TensorRT REQUIRED)

# Check the current platform to compiler 
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

# target_link_libraries(decodeplugin nvinfer cudart)

# Targets
add_executable(test_infer2features ${PROJECT_SOURCE_DIR}/main.cpp)
target_include_directories(test_infer2features PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(test_infer2features nvonnxparser)
target_link_libraries(test_infer2features PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_link_libraries(test_infer2features nvinfer)
target_link_libraries(test_infer2features cudart)
# target_link_libraries(test_infer2features decodeplugin)
target_link_libraries(test_infer2features ${OpenCV_LIBRARIES})
