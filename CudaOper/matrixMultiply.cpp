#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <driver_types.h>
#include <ctime>
#include "../DCT_Creator_Matrices.cpp"

__device__  void matrixMultiply(double *matrix1, double *matrix2, int window_size, double *result)
{
    double sum = 0;
    for(int i = 0; i < window_size; i++)
        for(int j = 0; j < window_size; j++) {
            for(int k = 0; k < window_size; k++){
                sum += *(matrix1+(window_size*i)+k) * (*(matrix2+(window_size*k)+j));
            }
            *(result+(window_size*i)+j) = sum;
            sum = 0;
        }
}

__global__ void DCT_filter(char *image, int* image_width, int *pixel_delay, double* DCT_Creator, double* DCT_Creator_T, int* window_size, double *threshold, char *result)
{
    int x = (blockIdx.x)* (*window_size) * (*pixel_delay) + (threadIdx.x);
    int y = (blockIdx.y)* (*window_size);

    double* image_block = new double[(*window_size)*(*window_size)];
    int p_dy = y;
    int p_dx = x;
    for (int k = 0; k < (*window_size); k++, p_dy++) {
        for (int t = 0, p_dx = x; t < (*window_size); t++, p_dx += (*pixel_delay)){
            image_block[((*window_size) * (k)) + (t)] = image[((*image_width) * p_dy) + p_dx];
        }
    }

    double* temp_block = new double[(*window_size)*(*window_size)];
    matrixMultiply(DCT_Creator, image_block, (*window_size), temp_block);
    matrixMultiply(temp_block, DCT_Creator_T, (*window_size), image_block);

    /*for(int i = 0; i < *window_size; i++)
    {
        for(int j = 0; j < *window_size; j++)
        {
            if(image_block[(*window_size * i) + j] > *threshold)
                image_block[(*window_size * i) + j] = 0;
        }
    }*/

    matrixMultiply(image_block, DCT_Creator, (*window_size), temp_block);
    matrixMultiply(DCT_Creator_T, temp_block, (*window_size), image_block);

    p_dy = y;

    for (int k = 0; k < (*window_size); k++, p_dy++) {
        for (int t = 0, p_dx = x; t < (*window_size); t++, p_dx += (*pixel_delay)) {
            result[((*image_width) * p_dy) + p_dx] = image_block[((*window_size) * (k)) + (t)];
            //result[((*image_width) * p_dy) + p_dx] = image[((*image_width) * p_dy) + p_dx];
        }
    }

    free(temp_block);
    free(image_block);
}


char* DCT_Filrer(char *image_, int width_, int heigth_, int channels_)
{
    int wind_size = 8;
    int image_width = width_;
    double threshold = 255*0.012;
    char *result = new char[image_width*heigth_];

    char *image_dev;
    double *DCT_creator_dev;
    double *DCT_creator_T_dev;
    double *threshold_dev;
    char *result_dev;
    int *window_size_dev;
    int *image_width_dev;
    int *pixel_delay_dev;


    cudaMalloc((void**)&image_dev, sizeof(char)*image_width*heigth_); // 256 is image size
    cudaMalloc((void**)&DCT_creator_dev, sizeof(double)*wind_size*wind_size);
    cudaMalloc((void**)&DCT_creator_T_dev, sizeof(double)*wind_size*wind_size);
    cudaMalloc((void**)&threshold_dev, sizeof(double));
    cudaMalloc((void**)&window_size_dev, sizeof(int)*1);
    cudaMalloc((void**)&image_width_dev, sizeof(int)*1);
    cudaMalloc((void**)&pixel_delay_dev, sizeof(int)*1);
    cudaMalloc((void**)&result_dev, sizeof(char)*image_width*heigth_);


    unsigned int start_time = clock();
    cudaMemcpyAsync(image_dev, image_, sizeof(char)*image_width*heigth_, cudaMemcpyHostToDevice);   //copy image to videomemory
    cudaMemcpy(threshold_dev, &threshold, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(window_size_dev, &wind_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(image_width_dev, &image_width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pixel_delay_dev, &channels_, sizeof(int), cudaMemcpyHostToDevice);

    switch(wind_size){   // copy DCT creator to video
        case 2:
            cudaMemcpyAsync(DCT_creator_dev, ImProcessing::DCT_Creator2, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(DCT_creator_T_dev, ImProcessing::DCT_Creator2_T, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            break;
        case 4:
            cudaMemcpyAsync(DCT_creator_dev, ImProcessing::DCT_Creator4, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(DCT_creator_T_dev, ImProcessing::DCT_Creator4_T, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            break;
        case 8:
            cudaMemcpyAsync(DCT_creator_dev, ImProcessing::DCT_Creator8, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(DCT_creator_T_dev, ImProcessing::DCT_Creator8_T, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            break;
        case 16:
            cudaMemcpyAsync(DCT_creator_dev, ImProcessing::DCT_Creator16, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(DCT_creator_T_dev, ImProcessing::DCT_Creator16_T, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            break;
        case 32:
            cudaMemcpyAsync(DCT_creator_dev, ImProcessing::DCT_Creator32, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(DCT_creator_T_dev, ImProcessing::DCT_Creator32_T, sizeof(double)*wind_size*wind_size, cudaMemcpyHostToDevice);
            break;
    }

    dim3 grid((width_)/(wind_size*channels_),heigth_/wind_size);

    DCT_filter<<<grid, channels_>>>(image_dev, image_width_dev, pixel_delay_dev, DCT_creator_dev, DCT_creator_T_dev, window_size_dev, threshold_dev, result_dev);

    cudaMemcpyAsync(result, result_dev, image_width * heigth_ * sizeof(char), cudaMemcpyDeviceToHost);

    unsigned int finish_time = clock();
    printf("%f \n", (float) (finish_time - start_time) / CLOCKS_PER_SEC);
    cudaFree(result_dev);
    cudaFree(image_dev);
    cudaFree(image_width_dev);
    cudaFree(threshold_dev);
    cudaFree(window_size_dev);
    cudaFree(DCT_creator_dev);
    cudaFree(DCT_creator_T_dev);

    return result;
}