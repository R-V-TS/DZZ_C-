//
// Created by rostyslav on 22.02.20.
//

#ifndef DZZ_IMAGEOPERATIONS_H
#define DZZ_IMAGEOPERATIONS_H

#include <cstdint>

uint8_t* normalizationIm(const uint16_t* image, const int length);

uint16_t maxOf(const uint16_t* image, const int length);

#endif //DZZ_IMAGEOPERATIONS_H
