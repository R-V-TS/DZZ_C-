#ifndef DZZ_FILTERS_H
#define DZZ_FILTERS_H

#include "Distort.h"
#include <cmath>

template <typename T>
void QuickSort(T* array, int start, int finish){
    T pivot_arr = array[finish];
    int i = start-1;

    for(int j = start; j <= finish-1; j++){
        if(array[j] < pivot_arr){
            i++;
            swap(&array[j], &array[i]);
        }
    }
    swap(&array[finish], &array[i+1]);

    if(start < i && i != finish-1){
        QuickSort(array, start, i);
    }
    if(i+2 < finish){
        QuickSort(array, i+2, finish);
    }
}

template <typename T>
T med(T* array, int length){
    QuickSort(array, 0, length-1);
    return array[(length/2)];
}

template <typename T>
void MedFilter(T* image, int width, int height, int block_size){

    auto* block = new T[block_size*block_size];

    for(int i = block_size/2; i < height-(block_size/2); i++){
        for(int j = block_size/2; j < width-(block_size/2); j++){
            for(int k = i - (block_size/2), x = 0; k <= i + block_size/2; k++, x++){
                for(int l = j - block_size/2, y = 0; l <= j + block_size/2; l++, y++){
                    block[(x * block_size) + y] = image[(k * width) + l];
                }
            }
            T median = med(block, block_size*block_size);
            image[(i*width) + j] = median;
        }
    }

    free(block);
}

template <typename T>
float local_mean(T* array, int length){
    float sum = 0;
    for(int i = 0; i < length; i++){
        sum += array[i];
    }
    return sum/length;
}

template <typename T>
float local_var(T* array, float mean, int length){
    float sum = 0;
    for(int i = 0; i < length; i++){
        sum += pow(array[i] - mean, 2);
    }
    return (sum/length);
}

template <typename T>
void LiFilter(T* image, int width, int height, int block_size, float SD){
    auto* block = new T[block_size*block_size];

    for(int i = block_size/2; i < height-(block_size/2); i++){
        for(int j = block_size/2; j < width-(block_size/2); j++){
            for(int k = i - (block_size/2), x = 0; k <= i + block_size/2; k++, x++){
                for(int l = j - block_size/2, y = 0; l <= j + block_size/2; l++, y++){
                    block[(x * block_size) + y] = image[(k * width) + l];
                }
            }
            float LM = local_mean(block, block_size*block_size);                // Local mean
            float LV = local_var(block, LM, block_size*block_size);             // Local variance
            T K = LV/(LV + SD);                                             // K = Lv / (Lv + SD)
            image[(i*width) + j] = LM + (K * (image[(i*width) + j] - LM));  // P = Lm + K * (P - Lm)
        }
    }

    free(block);
}


float* S_creator(uint16_t block_size){
    float* S = new float[block_size*block_size];

    float point_val = 0;

    for(int i = block_size/2, i_counter = 0; i >= 0; i--, i_counter++){
        for(int j = block_size/2, j_counter = 0; j >= 0; j--, j_counter++){
            point_val = sqrt(j_counter + i_counter);
            S[(i * block_size) + j] = point_val;
            S[(block_size - i - 1) * block_size + j] = point_val;
            S[(i * block_size) + (block_size - j - 1)] = point_val;
            S[((block_size - i - 1) * block_size) + (block_size - j - 1)] = point_val;
        }
    }

    /*for(int i = 0; i < block_size; i++){
        for(int j = 0; j < block_size; j++){
            printf("%f ", S[(i * block_size) + j]);
        }
        printf("\n");
    }*/

    return S;
}

template <typename T>
void FrostFilter(T* image, int width, int height, int wind_size){
    int block_size = wind_size;
    float D = 1, B = 0;
    float Y_ch = 0, Y_zn = 0, W;

    auto* block = new T[block_size*block_size];
    float* S = S_creator(wind_size);
    for(int i = block_size/2; i < height-(block_size/2); i++){
        for(int j = block_size/2; j < width-(block_size/2); j++){
            for(int k = i - (block_size/2), x = 0; k <= i + block_size/2; k++, x++){
                for(int l = j - block_size/2, y = 0; l <= j + block_size/2; l++, y++){
                    block[(x * block_size) + y] = image[(k * width) + l];
                }
            }
            float LM = local_mean(block, block_size*block_size);                // Local mean
            float LV = local_var(block, LM, block_size*block_size);             // Local variance
            B = D * (LV / (LM * LM));
            Y_ch = 0;
            Y_zn = 0;
            for(int z = 0; z < block_size*block_size; z++){
                W = exp(-B * S[z]);
                Y_ch += block[z] * W;
                Y_zn += W;
            }
            image[(i*width) + j] = uint8_t(Y_ch/Y_zn);
        }
    }

    free(block);
}

#endif //DZZ_FILTERS_H
