//
// Created by rostyslav on 05.03.20.
//
#include <cstdint>
#include "./../utils.h"
#include <string>

#ifndef CUDA_DZZ_SIMPLYFILTERS_H
#define CUDA_DZZ_SIMPLYFILTERS_H




void CudaFilter(Image *image_, unsigned short block_size, float SD, uint16_t im_bl_size, const std::string filterName);

#endif //CUDA_DZZ_SIMPLYFILTERS_H
