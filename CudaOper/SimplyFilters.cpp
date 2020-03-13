//
// Created by rostyslav on 05.03.20.
//

#include "SimplyFilters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ctime>
#include <cstdio>
#include <string>
#include <cmath>
#include <cstring>
#include "../DCT.h"



__device__ void swap(uint8_t* one, uint8_t* two){
    uint8_t tmp = *one;
    *one = *two;
    *two = tmp;
}

__device__ void QuickSort(uint8_t* block, unsigned int start, unsigned int finish){
    uint8_t pivot_arr = block[finish];
    int i = start-1;

    for(int j = start; j <= finish-1; j++){
        if(block[j] < pivot_arr){
            i++;
            swap(&block[j], &block[i]);
        }
    }
    swap(&block[finish], &block[i+1]);

    if(start < i && i != finish-1){
        QuickSort(block, start, i);
    }
    if(i+2 < finish){
        QuickSort(block, i+2, finish);
    }
}

__device__ void bubbleSort(uint8_t* array, unsigned int length){
    uint8_t tmp = 0;
    for(uint32_t i = 0; i < length-1; i++){
        for(uint32_t j = 0; j < length - i - 1; j++){
            if(array[j] > array[j+1]) {
                tmp = array[j];
                array[j] = array[j+1];
                array[j+1] = tmp;
            }
        }
    }
}

__device__ uint8_t median(uint8_t* block, unsigned int length){
    bubbleSort(block,length);
    //for(uint32_t z = 0; z < length; z++) printf("%i ", block[z]);
    //printf("\n");
    return block[length/2];
}

__device__ float localMean(uint8_t* array, uint32_t length){
    float sum = 0;
    for(uint32_t i = 0; i < length; i++){
        sum += array[i];
    }
    return sum/length;
}

__device__ void MultiplyMatrix(float *matrix1, float *matrix2, float *result, uint16_t window_size){
    float sum = 0;
    for(int i = 0; i < window_size; i++)
        for(int j = 0; j < window_size; j++) {
            for(int k = 0; k < window_size; k++){
                sum += *(matrix1+(window_size*i)+k) * (*(matrix2+(window_size*k)+j));
            }
            *(result+(window_size*i)+j) = sum;
            sum = 0;
        }
}

__device__ float localVariance(uint8_t* array, uint32_t length, float mean){
    float sum = 0;
    for(uint32_t i = 0; i < length; i++){
        sum += pow(array[i] - mean, 2);
    }
    return sum/length;
}

__device__ void padarray(const uint8_t* block, const uint32_t width_block, const uint32_t padding, const uint32_t width_pad_block, uint8_t* pad_block){
    uint32_t k = 0, t = 0, column_counter_top = 0, row_counter_top = 0, column_counter_bottom = 0, row_counter_bottom = 0;
    for(int i = 0; i < width_pad_block; i++){
        for(int j = padding; j < width_pad_block - padding; j++){
            if(i >= padding && i < padding + width_block && j >= padding && j < padding + width_block){
                pad_block[(i * width_pad_block) + j] = block[(k * width_block) + t];
                t++;
            } else if(i < padding) {
                pad_block[(i * width_pad_block) + j] = block[(row_counter_top * width_block) + column_counter_top];
                column_counter_top++;
                if(column_counter_top == width_block){
                    column_counter_top = 0;
                    row_counter_top++;
                }
                if(row_counter_top == width_block) row_counter_top = 0;
            } else if(i >= width_pad_block - padding){
                pad_block[(i * width_pad_block) + j] = block[(row_counter_bottom * width_block) + column_counter_bottom];
                column_counter_bottom++;
                if(column_counter_bottom == width_block){
                    column_counter_bottom = 0;
                    row_counter_bottom++;
                }
                if(row_counter_bottom == width_block) row_counter_bottom = 0;
            }
            if(t == width_block){
                t = 0;
                k++;
            }
        }
    }
    row_counter_bottom = 0, column_counter_bottom = padding;
    for(int i = 0; i < width_pad_block; i++, row_counter_bottom++){
        column_counter_bottom = padding;
        for(int j = padding; j >= 0; j--){
            //printf("%i %i | %i %i\n", i, j, row_counter_bottom, column_counter_bottom);
            pad_block[(i* width_pad_block) + j] = pad_block[(row_counter_bottom * width_pad_block) + column_counter_bottom];
            pad_block[(i* width_pad_block) + width_pad_block - j] = pad_block[(row_counter_bottom * width_pad_block) + ( width_pad_block - column_counter_bottom - 1)];
        }
    }
    /*for(int i = 0; i < width_pad_block; i++){
        for(int j = 0; j < width_pad_block; j++){
            printf("%3i ", pad_block[(i*width_pad_block) + j]);
        }
        printf("\n");
    }*/
}

__global__ void MedianFilter(uint8_t* image, unsigned int* width, unsigned short* window_size, const unsigned short* bl_width){
    unsigned int x_delay = (blockIdx.x) * (*bl_width);
    unsigned int y_delay = (blockIdx.y) * (*bl_width);
    unsigned int x_max = x_delay + *bl_width;
    unsigned int y_max = y_delay + *bl_width;


    unsigned int bl_pad = int(*window_size / 2);
    unsigned int size_block = *bl_width + bl_pad * 2;

    auto *block = new uint8_t[*bl_width * *bl_width];
    auto *padblock = new uint8_t[size_block * size_block];
    auto *filtblock = new uint8_t[*window_size * *window_size];


    for(uint32_t i = y_delay, i_b = 0; i < y_max; i++, i_b++){
        for(uint32_t j = x_delay, j_b = 0; j < x_max; j++, j_b++){
            block[(i_b * *bl_width) + j_b] = image[(i * *width) + j];
            //printf("%i %i -> %i \n", i, j, image[(i * *width) + j]);
        }
    }

    padarray(block, *bl_width, bl_pad, size_block, padblock);

    free(block);
    uint8_t med = 0;

    for(uint32_t i = bl_pad, i_im = y_delay; i < size_block - bl_pad; i++, i_im++){
        for(uint32_t j = bl_pad, j_im = x_delay; j < size_block - bl_pad; j++, j_im++){
            for(uint32_t y = 0, y_im = i - bl_pad; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - bl_pad; x < *window_size; x++, x_im++){
                    filtblock[(x * *window_size) + y] = padblock[(y_im * size_block) + x_im];
                    //printf("%i ", filtblock[(x * *window_size) + y]);
                }
                //printf("\n");
            }
            med = median(filtblock, *window_size * *window_size);
            //printf("%i \n\n", med);
            image[(i_im * *width) + j_im] = med;
        }
    }

    free(padblock);
    free(filtblock);
}

__global__ void LiFilter(uint8_t* image, uint32_t* width, uint16_t* window_size, uint16_t* bl_width, float* SD){
    unsigned int x_delay = (blockIdx.x) * (*bl_width);
    unsigned int y_delay = (blockIdx.y) * (*bl_width);
    unsigned int x_max = x_delay + *bl_width;
    unsigned int y_max = y_delay + *bl_width;


    unsigned int bl_pad = int(*window_size / 2);
    unsigned int size_block = *bl_width + bl_pad * 2;

    auto *block = new uint8_t[*bl_width * *bl_width];
    auto *padblock = new uint8_t[size_block * size_block];
    auto *filtblock = new uint8_t[*window_size * *window_size];


    for(uint32_t i = y_delay, i_b = 0; i < y_max; i++, i_b++){
        for(uint32_t j = x_delay, j_b = 0; j < x_max; j++, j_b++){
            block[(i_b * *bl_width) + j_b] = image[(i * *width) + j];
            //printf("%i %i -> %i \n", i, j, image[(i * *width) + j]);
        }
    }

    padarray(block, *bl_width, bl_pad, size_block, padblock);

    free(block);
    float LV = 0; // Local variance variable
    float LM = 0; // Local mean
    uint8_t K = 0;
    for(uint32_t i = bl_pad, i_im = y_delay; i < size_block - bl_pad; i++, i_im++){
        for(uint32_t j = bl_pad, j_im = x_delay; j < size_block - bl_pad; j++, j_im++){
            for(uint32_t y = 0, y_im = i - bl_pad; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - bl_pad; x < *window_size; x++, x_im++){
                    filtblock[(x * *window_size) + y] = padblock[(y_im * size_block) + x_im];
                    //printf("%i ", filtblock[(x * *window_size) + y]);
                }
                //printf("\n");
            }
            LM = localMean(filtblock, *window_size * *window_size);
            LV = localVariance(filtblock, *window_size * *window_size, LM);
            K = LV/(LV + *SD);
            //printf("%i \n\n", med);
            image[(i_im * *width) + j_im] = LM + (K * (image[(i_im * *width) + j_im] - LM));
        }
    }

    free(padblock);
    free(filtblock);
}

__global__ void FrostFilter(uint8_t* image, uint32_t* width, uint16_t* window_size, uint16_t* bl_width, float* S){
    unsigned int x_delay = (blockIdx.x) * (*bl_width);
    unsigned int y_delay = (blockIdx.y) * (*bl_width);
    unsigned int x_max = x_delay + *bl_width;
    unsigned int y_max = y_delay + *bl_width;

    unsigned int bl_pad = int(*window_size / 2);
    unsigned int size_block = *bl_width + bl_pad * 2;

    auto *block = new uint8_t[*bl_width * *bl_width];
    auto *padblock = new uint8_t[size_block * size_block];
    auto *filtblock = new uint8_t[*window_size * *window_size];


    for(uint32_t i = y_delay, i_b = 0; i < y_max; i++, i_b++){
        for(uint32_t j = x_delay, j_b = 0; j < x_max; j++, j_b++){
            block[(i_b * *bl_width) + j_b] = image[(i * *width) + j];
            //printf("%i %i -> %i \n", i, j, image[(i * *width) + j]);
        }
    }

    padarray(block, *bl_width, bl_pad, size_block, padblock);

    free(block);
    float LV = 0; // Local variance variable
    float LM = 0; // Local mean
    uint8_t K = 0;
    float Y_ch = 0, Y_zn = 0, W; // Variable for calculate Y
    for(uint32_t i = bl_pad, i_im = y_delay; i < size_block - bl_pad; i++, i_im++){
        for(uint32_t j = bl_pad, j_im = x_delay; j < size_block - bl_pad; j++, j_im++){
            for(uint32_t y = 0, y_im = i - bl_pad; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - bl_pad; x < *window_size; x++, x_im++){
                    filtblock[(x * *window_size) + y] = padblock[(y_im * size_block) + x_im];
                    //printf("%i ", filtblock[(x * *window_size) + y]);
                }
                //printf("\n");
            }
            LM = localMean(filtblock, *window_size * *window_size);
            LV = localVariance(filtblock, *window_size * *window_size, LM);
            K = bl_pad * (LV / (LM * LM));
            Y_ch = 0, Y_zn = 0;
            for(uint16_t z = 0; z < *window_size * *window_size; z++){
                W = exp(-K * S[z]);
                Y_ch += filtblock[z] * W;
                Y_zn += W;
            }
            //printf("%i \n\n", med);
            image[(i_im * *width) + j_im] = uint8_t(Y_ch/Y_zn);
        }
    }

    free(padblock);
    free(filtblock);
}

__global__ void DCT_Filter(uint8_t* image, uint32_t* width, uint16_t* window_size, uint16_t* bl_width, float* DCT_Creator, float* DCT_Creator_T, float* SD){

    unsigned int x_delay = (blockIdx.x) * (*bl_width);
    unsigned int y_delay = (blockIdx.y) * (*bl_width);
    unsigned int x_max = x_delay + *bl_width - *window_size;
    unsigned int y_max = y_delay + *bl_width - *window_size;

    float threshold = 2.7 * *SD;

    uint32_t *result_image = (uint32_t*)malloc(sizeof(uint32_t)* *bl_width * *bl_width);
    uint8_t *num_counter =  (uint8_t*)malloc(sizeof(uint8_t)* *bl_width * *bl_width);
    for(int i = 0; i < pow(*bl_width, 2); i++) {
        result_image[i] = 0;
        num_counter[i] = 0;
    }

    float *block = new float[*window_size * *window_size];
    float *temp = new float[*window_size * *window_size];

    for(int i = y_delay, i_r = 0; i <= y_max; i++, i_r++){
        for (int j = x_delay, j_r = 0; j <= x_max; j++, j_r++) {
            //printf("%i %i\n", i, j);
            //printf(" %i \n", *window_size);
            for(int z = 0; z < *window_size; z++){
                for(int l = 0; l < *window_size; l++){
                    block[(z* *window_size) + l] = (float)image[((i+z) * *width) + (j + l)];
                    //printf("%.1f ", block[(z* *window_size) + l]);
                }
                //printf("\n");
            }

            MultiplyMatrix(DCT_Creator, block, temp, *window_size);
            MultiplyMatrix(temp, DCT_Creator_T, block, *window_size);

            for(int z = 1; z < *window_size * *window_size; z++){
                if(fabsf(block[z]) <= threshold){
                    block[z] = 0;
                }
            }

            MultiplyMatrix(block, DCT_Creator, temp, *window_size);
            MultiplyMatrix(DCT_Creator_T, temp, block, *window_size);

            for(int z = 0; z < *window_size; z++){
                for(int l = 0; l < *window_size; l++){
                    result_image[((i_r+z) * *bl_width) + (j_r+l)] += (uint8_t) block[(z * *window_size) + l];
                    num_counter[((i_r+z) * *bl_width) + (j_r+l)] += 1;
                }
            }
        }
    }

    for(int i = y_delay, i_r = 0; i < y_max + *window_size; i++, i_r++) {
        for (int j = x_delay, j_r = 0; j < x_max + *window_size; j++, j_r++){
            uint8_t t = (uint8_t) (result_image[(i_r * *bl_width) + j_r] / (num_counter[(i_r * *bl_width) + j_r] != 0 ? num_counter[(i_r * *bl_width) + j_r] : 1));
            image[(i * *width) + j] = t;
        }
    }
    free(result_image);
    free(num_counter);
    free(block);
    free(temp);

}

__host__ float* S_creator(uint16_t block_size){
    float* S = new float[block_size*block_size];
    if(block_size == 3){
        S[0] = 1.4142;
        S[1] = 1;
        S[2] = 1.4142;
        S[3] = 1;
        S[4] = 0;
        S[5] = 1;
        S[6] = 1.4142;
        S[7] = 1;
        S[8] = 1.4142;
        return S;
    }
    return nullptr;
}

void CudaFilter(Image* image_, unsigned short block_size, float SD, const std::string filterName){

    unsigned short block_size_in_kernel = image_->width/16;
    uint8_t* image_dev;
    unsigned int* width_dev;
    unsigned int* height_dev;
    unsigned short* wind_size_dev;
    unsigned short* block_size_dev;
    float* noiseSD;
    float* S_block;
    float* S_block_dev;

    cudaMalloc((void**) &image_dev, sizeof(uint8_t)*image_->width*image_->height);
    cudaMalloc((void**) &width_dev, sizeof(unsigned int));
    cudaMalloc((void**) &height_dev, sizeof(unsigned int));
    cudaMalloc((void**) &wind_size_dev, sizeof(unsigned short));
    cudaMalloc((void**) &block_size_dev, sizeof(unsigned short));
    cudaMalloc((void**) &noiseSD, sizeof(float));

    unsigned int start_time = clock();
    cudaMemcpy(image_dev, image_->data, sizeof(uint8_t)*image_->width*image_->height, cudaMemcpyHostToDevice);
    cudaMemcpy(width_dev, &(image_->width), sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(height_dev, &(image_->height), sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(wind_size_dev, &block_size, sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(block_size_dev, &block_size_in_kernel, sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(noiseSD, &SD, sizeof(float), cudaMemcpyHostToDevice);
    unsigned int finish_time_copy = clock();
    // MedianFilter(uint8_t* image, unsigned int* width, unsigned short* window_size, unsigned short* bl_width, uint8_t* image_result)


    if(filterName == "Frost"){
        S_block = S_creator(block_size);
        if(S_block == nullptr){
            printf("Block is not support\n");
            return;
        }
        cudaMalloc((void**) &S_block_dev, sizeof(float) * block_size * block_size);
        cudaMemcpy(S_block_dev, S_block, sizeof(float) * block_size * block_size, cudaMemcpyHostToDevice);
    }


    if(filterName == "Median") MedianFilter<<<dim3(16, 16), 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev);
    else if(filterName == "Li") LiFilter<<<dim3(16, 16), 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev, noiseSD);
    else if(filterName == "Frost") FrostFilter<<<dim3(16, 16), 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev, S_block_dev);
    else if(filterName == "DCT"){
        float* DCT_creator;
        float* DCT_creator_T;

        cudaMalloc((void**) &DCT_creator, sizeof(float)*block_size*block_size);
        cudaMalloc((void**) &DCT_creator_T, sizeof(float)*block_size*block_size);

        if(block_size == 2){
            cudaMemcpy(DCT_creator, &DCT_Creator2[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
            cudaMemcpy(DCT_creator_T, &DCT_Creator2_T[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
        } else if(block_size == 4){
            cudaMemcpy(DCT_creator, &DCT_Creator4[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
            cudaMemcpy(DCT_creator_T, &DCT_Creator4_T[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
        } else if(block_size == 8){
            cudaMemcpy(DCT_creator, &DCT_Creator8[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
            cudaMemcpy(DCT_creator_T, &DCT_Creator8_T[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
        } else if(block_size == 16){
            cudaMemcpy(DCT_creator, &DCT_Creator16[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
            cudaMemcpy(DCT_creator_T, &DCT_Creator16_T[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
        } else if(block_size == 32){
            cudaMemcpy(DCT_creator, &DCT_Creator32[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
            cudaMemcpy(DCT_creator_T, &DCT_Creator32_T[0][0], sizeof(float)*block_size*block_size, cudaMemcpyHostToDevice);
        }
        DCT_Filter<<<dim3(16, 16), 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev, DCT_creator, DCT_creator_T, noiseSD);
    }

    else printf("Filter not found\n");

    unsigned int finish_time_kernel = clock();

    cudaMemcpy(image_->data, image_dev, sizeof(uint8_t)*image_->width*image_->height, cudaMemcpyDeviceToHost);
    unsigned int finish_time_copy_back = clock();

    printf("Копирование массива %f s\nРабота ядра %f s\nCopy back time %f s\n", (float)(finish_time_copy - start_time)/CLOCKS_PER_SEC, (float)(finish_time_kernel - finish_time_copy)/CLOCKS_PER_SEC, (float)(finish_time_copy_back - finish_time_kernel)/CLOCKS_PER_SEC);
    cudaFree(image_dev);
    cudaFree(width_dev);
    cudaFree(height_dev);
    cudaFree(wind_size_dev);
    cudaFree(block_size_dev);
}

