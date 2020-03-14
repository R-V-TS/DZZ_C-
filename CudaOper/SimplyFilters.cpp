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
    unsigned int y_delay = blockIdx.x * *bl_width;

    uint8_t *block = new uint8_t[*window_size * *window_size];

    uint8_t med = 0;

    uint16_t border = *window_size / 2;

    for(uint32_t i = border, i_im = y_delay + border; i < *bl_width - border; i++, i_im++){
        for(uint32_t j = border; j < *bl_width - border; j++){
            //printf("%i %i\n", i_im, j);
            for(uint32_t y = 0, y_im = i_im - border; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - border; x < *window_size; x++, x_im++){
                    block[(x * *window_size) + y] = image[(y_im * *bl_width) + x_im];
                    //printf("%i ", filtblock[(x * *window_size) + y]);
                }
                //printf("\n");
            }
            med = median(block, *window_size * *window_size);
            //printf("%i \n\n", med);
            image[(i_im * *width) + j] = med;
        }
    }

    free(block);
}

__global__ void LiFilter(uint8_t* image, uint32_t* width, uint16_t* window_size, uint16_t* bl_width, float* SD){
    unsigned int y_delay = blockIdx.x * *bl_width;

    uint8_t *block = new uint8_t[*window_size * *window_size];

    uint8_t med = 0;

    uint16_t border = *window_size / 2;

    float LV = 0; // Local variance variable
    float LM = 0; // Local mean
    uint8_t K = 0;

    for(uint32_t i = border, i_im = y_delay + border; i < *bl_width - border; i++, i_im++){
        for(uint32_t j = border; j < *bl_width - border; j++){
            //printf("%i %i\n", i_im, j);
            for(uint32_t y = 0, y_im = i_im - border; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - border; x < *window_size; x++, x_im++){
                    block[(x * *window_size) + y] = image[(y_im * *bl_width) + x_im];
                    //printf("%i ", filtblock[(x * *window_size) + y]);
                }
                //printf("\n");
            }
            LM = localMean(block, *window_size * *window_size);
            LV = localVariance(block, *window_size * *window_size, LM);
            K = LV/(LV + *SD);
            //printf("%i \n\n", med);
            image[(i_im * *width) + j] = LM + (K * (image[(i_im * *width) + j] - LM));;
        }
    }

    free(block);
}

__global__ void FrostFilter(uint8_t* image, uint32_t* width, uint16_t* window_size, uint16_t* bl_width, float* S){
    unsigned int y_delay = blockIdx.x * *bl_width;

    uint8_t *block = new uint8_t[*window_size * *window_size];

    uint8_t med = 0;

    uint16_t border = *window_size / 2;

    float LV = 0, Y_ch, Y_zn, W; // Local variance variable
    float LM = 0; // Local mean
    uint8_t K = 0;

    for(uint32_t i = border, i_im = y_delay + border; i < *bl_width - border; i++, i_im++){
        for(uint32_t j = border; j < *bl_width - border; j++){
            //printf("%i %i\n", i_im, j);
            for(uint32_t y = 0, y_im = i_im - border; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - border; x < *window_size; x++, x_im++){
                    block[(x * *window_size) + y] = image[(y_im * *bl_width) + x_im];
                    //printf("%i ", filtblock[(x * *window_size) + y]);
                }
                //printf("\n");
            }
            LM = localMean(block, *window_size * *window_size);
            LV = localVariance(block, *window_size * *window_size, LM);
            K = (LV / (LM * LM));
            Y_ch = 0, Y_zn = 0;
            for(uint16_t z = 0; z < *window_size * *window_size; z++){
                W = exp(-K * S[z]);
                Y_ch += block[z] * W;
                Y_zn += W;
            }
            //printf("%i \n\n", med);
            image[(i_im * *width) + j] = uint8_t(Y_ch/Y_zn);
        }
    }

    free(block);
}

__global__ void DCT_Filter(uint8_t* image, uint32_t* width, uint16_t* window_size, uint16_t* bl_width, float* DCT_Creator, float* DCT_Creator_T, float* SD){

    unsigned int y_delay = blockIdx.x * *bl_width;

    auto *block = new float[*window_size * *window_size];
    auto *temp = new float[*window_size * *window_size];

    float threshold = 2.7 * *SD;

    uint32_t *result_image = (uint32_t*)malloc(sizeof(uint32_t) * *bl_width * *bl_width);
    uint8_t *num_counter =  (uint8_t*)malloc(sizeof(uint8_t) * *bl_width * *bl_width);
    for(int i = 0; i < pow(*bl_width, 2); i++) {
        result_image[i] = 0;
        num_counter[i] = 0;
    }

    uint16_t border = *window_size / 2;

    for(uint32_t i = border, i_im = y_delay + border; i < *bl_width - border; i++, i_im++){
        for(uint32_t j = border; j < *bl_width - border; j++){
            for(uint32_t y = 0, y_im = i_im - border; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - border; x < *window_size; x++, x_im++){
                    block[(x * *window_size) + y] = float(image[(y_im * *bl_width) + x_im]);
                }
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

            for(uint32_t y = 0, y_im = i - border; y < *window_size; y++, y_im++){
                for(uint32_t x = 0, x_im = j - border; x < *window_size; x++, x_im++){
                    result_image[(y_im * *bl_width) + x_im] += uint32_t(block[(x * *window_size) + y]);
                    num_counter[(y_im * *bl_width) + x_im] += 1;
                }
            }
        }
    }

    for(int i = y_delay, i_r = 0; i_r < *bl_width; i++, i_r++) {
        for (int j = 0; j < *bl_width; j++){
            uint8_t t = uint8_t(float(result_image[(i_r * *bl_width) + j]) / float(num_counter[(i_r * *bl_width) + j] != 0 ? num_counter[(i_r * *bl_width) + j] : 1));
            image[(i * *bl_width) + j] = t;
        }
    }
    free(result_image);
    free(num_counter);
    free(block);
    free(temp);

}

__host__ float* S_creator(uint16_t block_size){
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

void CudaFilter(Image* image_, unsigned short block_size, float SD, uint16_t im_bl_size, const std::string filterName){

    unsigned short block_size_in_kernel = im_bl_size;
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


    if(filterName == "Median") MedianFilter<<<16*16, 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev);
    else if(filterName == "Li") LiFilter<<<16*16, 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev, noiseSD);
    else if(filterName == "Frost") FrostFilter<<<16*16, 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev, S_block_dev);
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
        DCT_Filter<<<16*16, 1>>>(image_dev, width_dev, wind_size_dev, block_size_dev, DCT_creator, DCT_creator_T, noiseSD);
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

