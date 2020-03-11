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

int main() {

    std::string imname_path = "../T42TXR_20190313T060631_B05.jp2";
    std::string imname = "T42TXR_20190313T060631_B05";
    const char *filename_metric = ("metrics_cpu" + imname + ".txt").c_str();
    const char *filename_time = ("timefile_cpu" + imname + ".txt").c_str();

    unsigned int start = 0;
    unsigned int finish = 0;

    GDALDataset *poDataset;
    GDALAllRegister();

    //Read image
    poDataset = (GDALDataset *) GDALOpen(imname_path.c_str(), GA_ReadOnly);

    int image_width = 512;
    int image_height = 512;

    GDALRasterBand *band = poDataset->GetRasterBand(1);

    int nXSize = band->GetXSize();
    int nYsize = band->GetYSize();

    //Type
    printf("DType = %i\n Size = %i, %i\n", band->GetRasterDataType(), nXSize, nYsize);

    auto* data = (uint16_t *) CPLMalloc(sizeof(uint16_t)*image_width*image_height);

    band->RasterIO( GF_Read, 512, 512, image_width, image_height,
                    data, image_width, image_height, band->GetRasterDataType(),
                    0, 0 );

    auto* norm_im = normalizationIm(data, image_height*image_width);

    saveImage(norm_im, image_width, image_height, "start_image"+imname+".png");

    Image ideal_image{};
    ideal_image.data = new uint8_t[image_width * image_height];
    memcpy(ideal_image.data, norm_im, sizeof(uint8_t) * image_width * image_height);
    ideal_image.width = image_width;
    ideal_image.height = image_height;

    Image noise_image{};
    noise_image.data = new uint8_t[image_width * image_height];
    noise_image.width = image_width;
    noise_image.height = image_height;

    Image image{};
    image.data = new uint8_t[image_width * image_height];
    image.width = image_width;
    image.height = image_height;

    FILE *out_metric;
    out_metric = fopen(filename_metric, "a+t");

    fprintf(out_metric, "Filter name \tSTD \tMSE \t PSNR\t PSNR-HVS\t PSNR-HVS-M\n");

    FILE *out_time;
    out_time = fopen(filename_time, "a+t");

    fprintf(out_time, "Filter name \tSTD \tExecute time\n");


    float noise_variance[5] = {5, 10, 15, 20, 25};

    for(auto i : noise_variance) {
        printf("STD = %f\n", i);
        memcpy(noise_image.data, ideal_image.data, sizeof(uint8_t) * image_width * image_height);

        printf("Add noise\n");
        AWGN(&noise_image, i, 0);
        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image

        saveImage(image.data, image_width, image_height, "noised_image_" + std::to_string(i) + "_" + imname + "_" + +".png");

        auto mse = MSE(&image, &ideal_image);
        auto psnr = PSNR(&ideal_image, &image);
        auto *psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

        fprintf(out_metric, "%11s \t%3.1f \t%4.2f \t %4.2f\t %5.3f\t %5.3f\n", "None", i, mse, psnr, psnrhvs[0], psnrhvs[1]);

        printf("Denoising \n");

        //Apply Median filter
        start = clock();
        MedFilter(image.data, image.width, image.height, 7);
        finish = clock();
        fprintf(out_time, "%11s \t%3.1f \t%4.2f\n", "Median", i, (float(finish-start)/CLOCKS_PER_SEC));
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

        fprintf(out_metric, "%11s \t%3.1f \t%4.2f \t %4.2f\t %5.3f\t %5.3f\n", "Median", i, mse, psnr, psnrhvs[0], psnrhvs[1]);

        saveImage(image.data, image_width, image_height, "Median_filter_" + std::to_string(i) + "_" + imname + ".png" );

        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image


        //Apply Li filter
        start = clock();
        LiFilter(image.data, image.width, image.height, 7, i);
        finish = clock();
        fprintf(out_time, "%11s \t%3.1f \t%4.2f\n", "Li", i, (float(finish-start)/CLOCKS_PER_SEC));
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

        fprintf(out_metric, "%11s \t%3.1f \t%4.2f \t %4.2f\t %5.3f\t %5.3f\n", "Li", i, mse, psnr, psnrhvs[0], psnrhvs[1]);

        saveImage(image.data, image_width, image_height, "Li_filter_" + std::to_string(i) + "_" + imname + ".png" );
        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image

        //Apply Frost filter
        start = clock();
        FrostFilter(image.data, image.width, image.height);
        finish = clock();
        fprintf(out_time, "%11s \t%3.1f \t%4.2f\n", "Frost", i, (float(finish-start)/CLOCKS_PER_SEC));
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

        fprintf(out_metric, "%11s \t%3.1f \t%4.2f \t %4.2f\t %5.3f\t %5.3f\n", "Frost", i, mse, psnr, psnrhvs[0], psnrhvs[1]);

        saveImage(image.data, image_width, image_height, "Frost_filter_" + std::to_string(i) + "_" + imname + ".png" );

        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image

        //Apply DCTBased filter
        start = clock();
        DCTBasedFilter(&image, i, 8);
        finish = clock();
        fprintf(out_time, "%11s \t%3.1f \t%4.2f\n", "DCTBased", i, (float(finish-start)/CLOCKS_PER_SEC));
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

        fprintf(out_metric, "%11s \t%3.1f \t%4.2f \t %4.2f\t %5.3f\t %5.3f\n", "DCTBased", i, mse, psnr, psnrhvs[0], psnrhvs[1]);

        saveImage(image.data, image_width, image_height, "DCTBased_filter_" + std::to_string(i) + "_" + imname + ".png" );
    }

    fclose(out_metric);
    fclose(out_time);

    free(data);
    free(ideal_image.data);
    free(image.data);
    free(noise_image.data);
    return 0;
}
