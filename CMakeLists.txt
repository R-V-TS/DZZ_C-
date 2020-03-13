cmake_minimum_required(VERSION 3.15)
project(DZZ)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgdal")
find_package(GDAL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

add_executable(DZZ main.cpp ImageOperations.cpp ImageOperations.h Distort.h Filters.h DCT.h utils.h DCT.cpp ImageQM.cpp ImageQM.h)

include_directories(include ${GDAL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Specify GDAL libraries to link your cpp executable target against
target_link_libraries(DZZ ${GDAL_LIBRARIES} ${OpenCV_LIBS} ${BOOST_LIBRARIES})

project(cuda_DZZ CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgdal")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(GDAL REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_INC_DIR /usr/local/cuda/include)


add_executable(cuda_DZZ CudaOper/main.cpp CudaOper/SimplyFilters.cpp CudaOper/SimplyFilters.h ImageOperations.cpp ImageQM.cpp DCT.cpp)
include_directories(cuda_DZZ ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${GDAL_INCLUDE_DIRS})
target_link_libraries(cuda_DZZ ${OpenCV_LIBS} ${GDAL_LIBRARIES})
set_source_files_properties(CudaOper/SimplyFilters.cpp PROPERTIES LANGUAGE CUDA)

target_compile_features(cuda_DZZ PUBLIC cxx_std_11)