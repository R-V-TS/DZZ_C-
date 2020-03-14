#include <iostream>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "ImageOperations.h"
#include "Distort.h"
#include "Filters.h"
#include "DCT.h"
#include "ImageQM.h"
#include <ctime>
#define TAB "\t"

template <typename T>
void displayImage(T* image, int width, int height, std::string im_name){
    cv::Mat im(width, height, CV_8U);

    for(int i = 0; i < width*height; i++){
        im.data[i] = image[i];
        //printf("%i ", data[i]);
    }


    cv::namedWindow("ImageShow", cv::WINDOW_NORMAL);
    cv::imshow("ImageShow", im);
    cv::imwrite(im_name, im);
    cv::resizeWindow("ImageShow", cv::Size(900, 900));
    cv::waitKey(0);
}

template <typename T>
void saveImage(T* image, int width, int height, std::string im_name){
    cv::Mat im(width, height, CV_8U);

    for(int i = 0; i < width*height; i++){
        im.data[i] = image[i];
        //printf("%i ", data[i]);
    }    
    cv::imwrite(im_name, im);
}

void save2file(std::fstream *stream, std::string filtName, float STD, int window_size, float MSE, float PSNR, float PSNRHVS, float PSNRHVSM, float time){
    *stream << filtName << "," << std::to_string(STD) << "," << std::to_string(window_size) << "," << std::to_string(MSE) << "," << std::to_string(PSNR) << "," << std::to_string(PSNRHVS) << "," << std::to_string(PSNRHVSM) << "," << std::to_string(time) << std::endl;
}


int main() {

    std::string imname_path = "../T42TXR_20190313T060631_B05.jp2";
    std::string imname = "T42TXR_20190313T060631_B05";
    std::string filename_metric = "metrics_cpu" + imname + ".csv";

    unsigned int start = 0;
    unsigned int finish = 0;

    GDALDataset *poDataset;
    GDALAllRegister();

    //Read image
    poDataset = (GDALDataset *) GDALOpen(imname_path.c_str(), GA_ReadOnly);

    int image_width = 512;
    int image_height = 512;
    int pad_size = 5;
    int im_with_pad = image_width + pad_size*2;

    GDALRasterBand *band = poDataset->GetRasterBand(1);

    int nXSize = band->GetXSize();
    int nYsize = band->GetYSize();

    //Type
    printf("DType = %i\n Size = %i, %i\n", band->GetRasterDataType(), nXSize, nYsize);

    auto* data = (uint16_t *) CPLMalloc(sizeof(uint16_t)*im_with_pad*im_with_pad);

    band->RasterIO( GF_Read, 512-pad_size, 512-pad_size, im_with_pad, im_with_pad,
                    data, im_with_pad, im_with_pad, band->GetRasterDataType(),
                    0, 0 );

    auto* norm_im = normalizationIm(data, im_with_pad*im_with_pad);

    //saveImage(norm_im, image_width, image_height, "start_image"+imname+".png");

    Image ideal_image{};
    ideal_image.data = new uint8_t[im_with_pad*im_with_pad];
    memcpy(ideal_image.data, norm_im, sizeof(uint8_t) * im_with_pad * im_with_pad);
    ideal_image.width = im_with_pad;
    ideal_image.height = im_with_pad;

    Image noise_image{};
    noise_image.data = new uint8_t[im_with_pad*im_with_pad];
    noise_image.width = im_with_pad;
    noise_image.height = im_with_pad;

    Image filter_image{};
    filter_image.data = new uint8_t[im_with_pad*im_with_pad];
    filter_image.width = im_with_pad;
    filter_image.height = im_with_pad;


    Image ideal_without_pad{};
    ideal_without_pad.data = new uint8_t[image_height*image_width];
    for(int i = 0; i < image_width; i++){
        for(int j = 0; j < image_height; j++){
            ideal_without_pad.data[(i * image_width) + j] = ideal_image.data[((i + pad_size) * im_with_pad) + (j + pad_size)];
        }
    }

    ideal_without_pad.width = image_width;
    ideal_without_pad.height = image_height;

    Image image{};
    image.data = new uint8_t[image_width * image_height];
    image.width = image_width;
    image.height = image_height;

    std::fstream outFile;
    outFile.open(filename_metric, std::ios::out);

    outFile << "Filter_name" << "," << "Noise STD" << "," << "Window size" << "," << "MSE" << "," << "PSNR" << "," << "PSNRHVS" << "," << "PSNRHVSM" << "," << "Execute time" << std::endl;

    float noise_variance[5] = {5, 10, 15, 20, 25};
    int window_sizes[4] = {3,5,7,9};

    for(auto i : noise_variance) {
        printf("STD = %f\n", i);
        memcpy(noise_image.data, ideal_image.data, sizeof(uint8_t) * im_with_pad * im_with_pad);

        printf("Add noise\n");
        AWGN(&noise_image, i, 0);

        auto mse = MSE(&image, &ideal_without_pad);
        auto psnr = PSNR(&ideal_image, &ideal_without_pad);
        auto *psnrhvs = PSNRHVSM(&image, &ideal_without_pad);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

        printf("Denoising \n");

        //Apply Median filter
        for(auto wind_s: window_sizes) {
            memcpy(filter_image.data, noise_image.data, sizeof(uint8_t) * im_with_pad * im_with_pad);
            start = clock();
            MedFilter(filter_image.data, im_with_pad, im_with_pad, wind_s);
            finish = clock();

            //Delete paddings
            for(int i = 0; i < image_width; i++){
                for(int j = 0; j < image_height; j++){
                    image.data[(i * image_width) + j] = filter_image.data[((i + pad_size) * im_with_pad) + (j + pad_size)];
                }
            }

            auto mse = MSE(&image, &ideal_without_pad);
            auto psnr = PSNR(&ideal_image, &ideal_without_pad);
            auto *psnrhvs = PSNRHVSM(&image, &ideal_without_pad);
            printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
            save2file(&outFile, "Median", i, wind_s, mse, psnr, psnrhvs[0], psnrhvs[1],
                      (float(finish - start) / CLOCKS_PER_SEC));
            saveImage(image.data, image_width, image_height,
                      "Median_filter_" + std::to_string(wind_s) + "_" + std::to_string(i) + "_" + imname + ".png");
        }

        //Apply Li filter
        for(auto wind_s: window_sizes) {
            memcpy(filter_image.data, noise_image.data, sizeof(uint8_t) * im_with_pad * im_with_pad);
            start = clock();
            LiFilter(filter_image.data, im_with_pad, im_with_pad, wind_s, i);
            finish = clock();

            //Delete paddings
            for(int i = 0; i < image_width; i++){
                for(int j = 0; j < image_height; j++){
                    image.data[(i * image_width) + j] = filter_image.data[((i + pad_size) * im_with_pad) + (j + pad_size)];
                }
            }

            auto mse = MSE(&image, &ideal_without_pad);
            auto psnr = PSNR(&ideal_image, &ideal_without_pad);
            auto *psnrhvs = PSNRHVSM(&image, &ideal_without_pad);
            printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
            save2file(&outFile, "Li", i, wind_s, mse, psnr, psnrhvs[0], psnrhvs[1],
                      (float(finish - start) / CLOCKS_PER_SEC));
            saveImage(image.data, image_width, image_height,
                      "Li_filter_" + std::to_string(wind_s) + "_" + std::to_string(i) + "_" + imname + ".png");
        }

        //Apply Frost filter
        for(auto wind_s: window_sizes) {
            memcpy(filter_image.data, noise_image.data, sizeof(uint8_t) * im_with_pad * im_with_pad);
            start = clock();
            FrostFilter(filter_image.data, im_with_pad, im_with_pad, wind_s);
            finish = clock();

            //Delete paddings
            for(int i = 0; i < image_width; i++){
                for(int j = 0; j < image_height; j++){
                    image.data[(i * image_width) + j] = filter_image.data[((i + pad_size) * im_with_pad) + (j + pad_size)];
                }
            }

            auto mse = MSE(&image, &ideal_without_pad);
            auto psnr = PSNR(&ideal_image, &ideal_without_pad);
            auto *psnrhvs = PSNRHVSM(&image, &ideal_without_pad);
            printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
            save2file(&outFile, "Frost", i, wind_s, mse, psnr, psnrhvs[0], psnrhvs[1],
                      (float(finish - start) / CLOCKS_PER_SEC));
            saveImage(image.data, image_width, image_height,
                      "Frost_filter_" + std::to_string(wind_s) + "_" + std::to_string(i) + "_" + imname + ".png");
        }

        //Apply DCTBased filter
        memcpy(filter_image.data, noise_image.data, sizeof(uint8_t) * im_with_pad * im_with_pad);
        start = clock();
        DCTBasedFilter(&filter_image, i, 8);
        finish = clock();

        //Delete paddings
        for(int i = 0; i < image_width; i++){
            for(int j = 0; j < image_height; j++){
                image.data[(i * image_width) + j] = filter_image.data[((i + pad_size) * im_with_pad) + (j + pad_size)];
            }
        }

        mse = MSE(&image, &ideal_without_pad);
        psnr = PSNR(&ideal_image, &ideal_without_pad);
        psnrhvs = PSNRHVSM(&image, &ideal_without_pad);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
        save2file(&outFile, "Frost", i, 8, mse, psnr, psnrhvs[0], psnrhvs[1],
                  (float(finish - start) / CLOCKS_PER_SEC));
        saveImage(image.data, image_width, image_height,
                  "Frost_filter_" + std::to_string(8) + "_" + std::to_string(i) + "_" + imname + ".png");
    }

    outFile.close();

    free(data);
    free(ideal_image.data);
    free(image.data);
    free(noise_image.data);
    free(ideal_without_pad.data);
    return 0;
}
