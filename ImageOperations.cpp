#include "ImageOperations.h"


uint16_t maxOf(const uint16_t* image, const int length){
    uint16_t max=0;
    for(int i=0; i<length; i++)
        if(image[i]>max) max = image[i];
    return max;
}

uint8_t* normalizationIm(const uint16_t* image, const int length){
    auto* out = new uint8_t[length];
    uint16_t max = maxOf(image, length);
    for(int i=0; i < length; i++){
        out[i] = uint8_t ((float(image[i])/float(max))*255);
    }
    return out;
}