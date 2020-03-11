//
// Created by rostyslav on 05.03.20.
//

#include <cstdint>

#ifndef CUDA_DZZ_UTILS_H
#define CUDA_DZZ_UTILS_H

struct Image{
    uint8_t* data;
    unsigned int width;
    unsigned int height;
};

#endif //CUDA_DZZ_UTILS_H
