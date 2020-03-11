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


int main() {

    GDALDataset *poDataset;
    GDALAllRegister();
    //Read image
    poDataset = (GDALDataset *) GDALOpen("../T42TXR_20190313T060631_B05.jp2", GA_ReadOnly);

    GDALRasterBand *band = poDataset->GetRasterBand(1);
    int width = band->GetXSize();
    int height = band->GetYSize();
    int dtype = band->GetRasterDataType();

    int nXSize = band->GetXSize();
    int nYsize = band->GetYSize();

    //Type
    printf("DType = %i\n Size = %i, %i\n", band->GetRasterDataType(), nXSize, nYsize);

    uint16_t* data = (uint16_t *) CPLMalloc(sizeof(uint16_t)*nXSize*nYsize);
    band->RasterIO( GF_Read, 0, 0, nXSize, nYsize,
                    data, nXSize, nYsize, band->GetRasterDataType(),
                    0, 0 );

    auto* norm_im = normalizationIm(data, nXSize*nYsize);
    int im_size= 1700;

    Image img;
    img.data = new uint8_t[im_size * im_size];
    img.width = im_size;
    img.height = im_size;

    for(int i = 0; i < im_size; i++){
        for(int j = 0; j < im_size; j++){
            img.data[(i*im_size) + j] = norm_im[(i * nXSize) + j];
        }
    }

    AWGN(img.data, img.width*img.height, 35, 0);

    CudaFilter(&img, 3, 35, "Frost");


    displayImage(img.data, img.width, img.height);

    free(data);
    return 0;
}
