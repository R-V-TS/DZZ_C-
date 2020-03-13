//
// Created by rostyslav on 25.02.20.
//
#include <cstdint>
#include "utils.h"

#ifndef DZZ_DCT_H
#define DZZ_DCT_H

extern float DCT_Creator2[2][2];

extern float DCT_Creator2_T[2][2];

extern float DCT_Creator4[4][4];

extern float DCT_Creator4_T[4][4];

extern float DCT_Creator8[8][8];

extern float DCT_Creator8_T[8][8];

extern float DCT_Creator16[16][16];

extern float DCT_Creator16_T[16][16];

extern float DCT_Creator32[32][32];

extern float DCT_Creator32_T[32][32];

void getImageBlock(uint8_t* image, int i_, int j_, int image_width, int window_size, float* block);
void getImageBlock(float* image, int i_, int j_, int image_width, int window_size, float* block);
float* DCT(float *block, int block_size);
void DCTBasedFilter(Image *image, float noise_variance, int window_size);

#endif //DZZ_DCT_H
