#include <iostream>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "./../ImageOperations.h"
#include "./../utils.h"
#include "./../Distort.h"
#include "SimplyFilters.h"
#include "../ImageQM.h"
#include <ctime>


template <typename T>
void displayImage(T* image, int width, int height){
    cv::Mat im(width, height, CV_8U);

    for(int i = 0; i < width*height; i++){
        im.data[i] = image[i];
        //printf("%i ", data[i]);
    }


    cv::namedWindow("ImageShow", cv::WINDOW_NORMAL);
    cv::imshow("ImageShow", im);
    cv::resizeWindow("ImageShow", cv::Size(900, 900));
    cv::waitKey(0);
    cv::imwrite("result.png", im);
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

void save2file(std::fstream *stream, std::string filtName, float STD, float MSE, float PSNR, float PSNRHVS, float PSNRHVSM, float time){
    *stream << filtName << "," << std::to_string(STD) << "," << std::to_string(MSE) << "," << std::to_string(PSNR) << "," << std::to_string(PSNRHVS) << "," << std::to_string(PSNRHVSM) << "," << std::to_string(time) << std::endl;
}

uint8_t* Image2Stack(uint8_t* image_pixel, uint32_t width, uint32_t height, uint32_t pad_width, uint32_t block_width){
    uint32_t num_of_block = (width/block_width) * (height/block_width);
    uint8_t *pad_image_stack = new uint8_t[(block_width + pad_width * 2) * (num_of_block * (block_width + (pad_width * 2)))];

    uint32_t top_tab = 0;
    int count = 0;

    for(int i = pad_width; i < height + pad_width; i+= block_width){
        for(int j = pad_width; j < width + pad_width; j+= block_width){
            for(int z = i - pad_width, z_bl = 0; z_bl < block_width + (pad_width * 2); z++, z_bl++){
                count = 0;
                for(int l = j - pad_width, l_bl = 0; l_bl < block_width + (pad_width*2); l++, l_bl++){
                    pad_image_stack[(top_tab * (block_width + pad_width*2)) + l_bl] = image_pixel[(z * (width + (2*pad_width))) + l];
                    //printf("%i ", pad_image_stack[(top_tab * (block_width + pad_width*2)) + l_bl]);
                    count++;
                }
                top_tab++;
                //printf("\n");
            }
            //printf("\n", top_tab, count);
        }
    }
    return pad_image_stack;
}

uint8_t* stack2Image(uint8_t* stack, uint32_t width, uint32_t height, uint32_t pad_width, uint32_t block_width){
    uint8_t *image = new uint8_t[width * height];

    uint32_t top_tab = pad_width;
    int count = 0;

    for(int i = 0; i < height; i += block_width){
        for(int j = 0; j < width; j += block_width){
            for(int z = pad_width, z_bl = 0; z_bl < block_width; z++, z_bl++){
                count = 0;
                for(int l = pad_width, l_bl = 0; l_bl < block_width; l++, l_bl++){
                    image[(z_bl + i) * (width) + (l_bl + j)] = stack[(top_tab * (block_width + pad_width*2)) + l];
                    //printf("%i ", image[(z_bl + i) * (width) + l_bl]);
                    count++;
                }
                top_tab++;
                //printf("\n");
            }
            top_tab += 2*pad_width;
            //printf("%i %i \n", top_tab, count);
        }
    }
    return image;
}

int main() {

    std::string imname_path = "../T42TXR_20190313T060631_B05.jp2";
    std::string imname = "T42TXR_20190313T060631_B05";
    std::string filename_metric = "metrics_gpu" + imname + ".csv";

    unsigned int start = 0;
    unsigned int finish = 0;

    GDALDataset *poDataset;
    GDALAllRegister();

    //Read image
    poDataset = (GDALDataset *) GDALOpen(imname_path.c_str(), GA_ReadOnly);

    int image_width = 512;
    int image_height = 512;
    int block_pad = 5;

    GDALRasterBand *band = poDataset->GetRasterBand(1);

    int nXSize = band->GetXSize();
    int nYsize = band->GetYSize();

    //Type
    printf("DType = %i\n Size = %i, %i\n", band->GetRasterDataType(), nXSize, nYsize);

    auto* data = (uint16_t *) CPLMalloc(sizeof(uint16_t)*(image_width + (block_pad * 2))*(image_width + (block_pad * 2)));

    band->RasterIO( GF_Read, 512 - block_pad, 512 - block_pad, image_width + (block_pad * 2), image_height + (block_pad * 2),
                    data, image_width + (block_pad * 2), image_height + (block_pad * 2), band->GetRasterDataType(),
                    0, 0 );

    uint8_t* norm_im = normalizationIm(data, (image_width + (block_pad * 2))*(image_height + (block_pad * 2)));
    //displayImage(norm_im, image_width + (block_pad * 2), image_height + (block_pad * 2));

    uint8_t * im_stack = Image2Stack(norm_im, image_width, image_height, block_pad, 32);

    free(norm_im);


    Image ideal_image{};
    ideal_image.data = new uint8_t[(image_width + 2 * (block_pad)) * (image_height + 2 * (block_pad))];
    memcpy(ideal_image.data, im_stack, sizeof(uint8_t) * (image_width + 2 * (block_pad)) * (image_height + 2 * (block_pad)));
    ideal_image.width = image_width + 2*block_pad;
    ideal_image.height = image_height + 2*block_pad;

    Image noise_image{};
    noise_image.data = new uint8_t[(image_width + 2 * (block_pad)) * (image_height + 2 * (block_pad))];
    noise_image.width = image_width;
    noise_image.height = image_height;

    Image image{};
    image.data = new uint8_t[(image_width + 2 * (block_pad)) * (image_height + 2 * (block_pad))];
    image.width = image_width;
    image.height = image_height;

    memcpy(ideal_image.data, im_stack, sizeof(uint8_t) * (image_width + 2 * (block_pad)) * (image_height + 2 * (block_pad)));

    printf("Add noise\n");
    AWGN(&noise_image, 15, 0);
    memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image

   // saveImage(image.data, image_width, image_height, "noised_image_" + std::to_string(15) + "_" + imname + "_" + +".png");

    auto* image = stack2Image(im_stack, image_width, image_height, block_pad, 32);

    displayImage(image, image_width, image_height);

    auto mse = MSE(&image, &ideal_image);
    auto psnr = PSNR(&ideal_image, &image);
    auto *psnrhvs = PSNRHVSM(&image, &ideal_image);
    printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);

    //std::fstream outFile;
    //outFile.open(filename_metric, std::ios::out);

    //outFile << "Filter_name" << "," << "Noise STD" << "," << "MSE" << "," << "PSNR" << "," << "PSNRHVS" << "," << "PSNRHVSM" << "," << "Execute time" << std::endl;

    /*float noise_variance[5] = {5, 10, 15, 20, 25};

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

        save2file(&outFile, "None", i, mse, psnr, psnrhvs[0], psnrhvs[1], 0);


        printf("Denoising \n");

        //Apply Median filter
        start = clock();
        CudaFilter(&image, 7, i, "Median");
        finish = clock();
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
        save2file(&outFile, "Median", i, mse, psnr, psnrhvs[0], psnrhvs[1], (float(finish-start)/CLOCKS_PER_SEC));
        saveImage(image.data, image_width, image_height, "Median_filter_" + std::to_string(i) + "_" + imname + ".png" );

        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image


        //Apply Li filter
        start = clock();
        CudaFilter(&image, 7, i, "Li");
        finish = clock();
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
        save2file(&outFile, "Li", i, mse, psnr, psnrhvs[0], psnrhvs[1], (float(finish-start)/CLOCKS_PER_SEC));
        saveImage(image.data, image_width, image_height, "Li_filter_" + std::to_string(i) + "_" + imname + ".png" );
        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image

        //Apply Frost filter
        start = clock();
        CudaFilter(&image, 3, i, "Frost");
        finish = clock();
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
        save2file(&outFile, "Frost", i, mse, psnr, psnrhvs[0], psnrhvs[1], (float(finish-start)/CLOCKS_PER_SEC));
        saveImage(image.data, image_width, image_height, "Frost_filter_" + std::to_string(i) + "_" + imname + ".png" );

        memcpy(image.data, noise_image.data, sizeof(uint8_t) * image_width * image_height); // Copy image

        //Apply DCTBased filter
        start = clock();
        CudaFilter(&image, 8, i, "DCT");
        finish = clock();
        mse = MSE(&image, &ideal_image);
        psnr = PSNR(&ideal_image, &image);
        psnrhvs = PSNRHVSM(&image, &ideal_image);
        printf("MSE = %f\nPSNR = %f\nPSNRHVS = %f\nPSNRHVSM = %f\n", mse, psnr, psnrhvs[0], psnrhvs[1]);
        save2file(&outFile, "DCTbased", i, mse, psnr, psnrhvs[0], psnrhvs[1], (float(finish-start)/CLOCKS_PER_SEC));

        saveImage(image.data, image_width, image_height, "DCTBased_filter_" + std::to_string(i) + "_" + imname + ".png" );
    }*/

   // outFile.close();



    /*free(data);
    free(ideal_image.data);
    free(image.data);
    free(noise_image.data);*/
    //free(im_stack);
    return 0;
}
