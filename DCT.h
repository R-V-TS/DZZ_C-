//
// Created by rostyslav on 25.02.20.
//
#include <cstdint>
#include "utils.h"

#ifndef DZZ_DCT_H
#define DZZ_DCT_H

void getImageBlock(uint8_t* image, int i_, int j_, int image_width, int window_size, float* block);
void getImageBlock(float* image, int i_, int j_, int image_width, int window_size, float* block);
float* DCT(float *block, int block_size);
void DCTBasedFilter(Image *image, float noise_variance, int window_size);

#endif //DZZ_DCT_H
