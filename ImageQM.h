//
// Created by rostyslav on 10.03.20.
//

#ifndef CUDA_DZZ_IMAGEQM_H
#define CUDA_DZZ_IMAGEQM_H

#include "utils.h"

float mean(float* im_block, int length);

float variance(float* im_block, int length);

double MSE(Image *imageQ, Image *imageP);

float PSNR(Image *imageP, Image *imageQ);

float maskeff(float* image, float* DCT_im, int window_size);

float* PSNRHVSM(Image *imageP, Image *imageQ);

#endif //CUDA_DZZ_IMAGEQM_H
