#ifndef DZZ_DISTORT_H
#define DZZ_DISTORT_H
#include <random>
#include <cmath>
#include <vector>
#include <ctime>
#include <thread>
#include "utils.h"

const float pi = 3.1415;

cv::Mat Array2Mat(const uint16_t* array, int height, int width){
    cv::Mat outMat(width, height, CV_16U);
    for(int i = 0; i < height*width; i++){
        outMat.data[i] = array[i];
    }
    return outMat;
}

cv::Mat Array2Mat(const float* array, int height, int width){
    cv::Mat outMat(width, height, CV_32F);
    for(int i = 0; i < height*width; i++){
        outMat.data[i] = array[i];
    }
    return outMat;
}

uint16_t* Mat2Array(const cv::Mat InMat, int height, int width){
    auto* outArr = new uint16_t[width*height];
    for(int i = 0; i < width*height; i++){
        outArr[i] = InMat.data[i];
    }
    return outArr;
}


template <typename T>
float stdCalc(T* image, int length){
    float sum = 0, mean = 0, stdI = 0;
    for(int i = 0; i < length; i++){
        sum += image[i];
    }
    mean = sum/length;
    for(int i = 0; i < length; i++){
        stdI += std::pow(image[i] - mean, 2);
    }
    return std::sqrt(stdI/length);
}

void meshgrid(float* x_in, float* y_in, int width, int height, float* x_out, float* y_out){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            x_out[(i*width) + j] = x_in[j];
            y_out[(i*width) + j] = y_in[i];
        }
    }
}

template <typename T>
float* ascn2D_fft_gen(T* imageAWGN, int width, int height, float gsigma){
    printf("Generate FFT ASCN\n");
    auto* ASCN = new float[width*height];
    auto* x = new float[width];
    auto* y = new float[height];

    for(int i = -width/2, j = 0; i < width/2; i++, j++){
        x[j] = i;
    }
    for(int i = -height/2, j = 0; i < height/2; i++, j++){
        y[j] = i;
    }

    auto* x_mesh = new float[width*height];
    auto* y_mesh = new float[height*width];
    meshgrid(x, y, width, height, x_mesh, y_mesh);
    auto* G = new float[width*height];
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            G[(i*width)+j] = std::exp(-pi * (pow(x_mesh[(i*width)+j], 2) + pow(y_mesh[(i*width)+j], 2))/(2*pow(gsigma, 2)));
        }
    }
    free(x); free(y); free(x_mesh); free(y_mesh);

    int length = width*height;

    cv::Mat Float_Mat_G, Float_Mat_AWGN;
    Float_Mat_G = Array2Mat(G, height, width);
    Float_Mat_AWGN = Array2Mat(imageAWGN, height, width);
    free(G);

    cv::Mat FT_G, FT_AWGN;
    cv::dft(Float_Mat_G, FT_G, cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
    cv::dft(Float_Mat_AWGN, FT_AWGN, cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);

    for(int i = 0; i < width*height; i++){
        FT_G.data[i] += FT_AWGN.data[i];
    }

    cv::Mat UintMat;
    cv::Mat IFT;
    cv::dft(FT_G, IFT, cv::DCT_INVERSE|cv::DFT_REAL_OUTPUT);
    IFT.convertTo(UintMat, CV_32F);

    for(int i = 0; i < length; i++){
        ASCN[i] = UintMat.data[i];
    }

    return ASCN;
}


double* getReyleighNum(float scale, int length){
    printf("Generate Reyleigh dist array\n");
    auto* noise = new double[length];
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0,1);
    double num = 0;
    for(int i = 0; i < length; i++){
        num = distribution(generator);
        noise[i] = sqrt(-2*(scale*scale)*std::log10(num));
    }

    return noise;
}

void AWGN(Image *image, const float sigma, const float mu){
    std::default_random_engine gen;
    std::normal_distribution<double> d{mu, sigma};

    int stars = 100;
    int nrols = image->height*image->width;
    int p[20] = {};

    for(int i=0; i<image->width*image->height;i++){
        int num = int(d(gen));
        if(image->data[i] + num > 255) image->data[i] = 255;
        else if(image->data[i] + num < 0) image->data[i] = 0;
        else image->data[i] += num;
        if(num < 10 && num >= -10) p[num+10]++;
    }

    /*for(int i = 0; i < 20; i++){
        printf("%i %s\n", i-10, std::string(p[i]*stars/nrols, '*').c_str());
    }*/
}



template <typename T>
void ASCN(T* image, const int width, const int height, const float nsigma, const float gsigma){
    int length = width*height;
    float* random_noise = new float[length];
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};
    for(int i = 0; i < length; i++)
        random_noise[i] = d(gen);
    float* ascn_noise = ascn2D_fft_gen(random_noise, width, height, gsigma);
    for(int i = 0; i < length; i++){
        image[i] += nsigma * ascn_noise[i];
    }
}

template <typename T>
void Mult(T* image, const int length, const int looks){
    float k = 0.8;
    auto* noise = new double[length];
    for(int l = 0; l < looks; l++){
        double *noise_gen = getReyleighNum(k, length);
        for(int i = 0; i < length; i++) {
            noise[i] += noise_gen[i];
        }
        free(noise_gen);
    }

    for(int i = 0; i < length; i++){
        image[i] *= std::round(noise[i]/looks);
    }
    free(noise);
}

template <typename T>
void swap(T* a, T* b){
    T c = *a;
    *a = *b;
    *b = c;
}

template <typename T>
void QuickSort(T* array, int* args, int start, int finish){
    T pivot_arr = array[finish];
    int pivot_arg = args[finish];
    int i = start-1;

    for(int j = start; j <= finish-1; j++){
        if(array[j] < pivot_arr){
            i++;
            swap(&array[j], &array[i]);
            swap(&args[j], &args[i]);
        }
    }

    int tmp = array[i+1];
    array[i+1] = array[i];
    array[i] = tmp;

    tmp = args[i+1];
    args[i+1] = args[i];
    args[i] = tmp;

    if(start < i && i != finish-1){
        QuickSort(array, args, start, i);
    }
    if(i+2 < finish){
        QuickSort(array, args, i+2, finish);
    }
}

template <typename T>
int* argsort(T* image, int length){
    printf("Sort Array\n");
    auto* args = new int[length];
    auto* aws = new T[length];
    for(int i = 0; i < length; i++){
        args[i] = i;
        aws[i] = image[i];
    }

    QuickSort(aws, args, 0, length-1);

    delete [] aws;
    return args;
}

template <typename T>
void Speckle(T* image, const int width, const int height, const int looks, const float gsigma){
    int length = width*height;
    float k = 0.8;
    auto* noise = new float[length];

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};
    for(int l = 0; l < looks; l++){
        printf("Looks #%i\n", l);
        for(int i = 0; i < length; i++)
            noise[i] = d(gen);
        float* ascn_noise = ascn2D_fft_gen(noise, width, height, gsigma);
        double* reyleigth_noise = getReyleighNum(k, length);
        auto* ascn_arg = argsort(ascn_noise, length);
        auto* releigh_arg = argsort(reyleigth_noise, length);
        for(int z = 0; z < length; z++)
        {
            ascn_noise[ascn_arg[z]] = reyleigth_noise[releigh_arg[z]];
            noise[z] = ascn_noise[ascn_arg[z]];
        }
    }
    for(int i = 0; i < length; i++)
    {
        image[i] *= (noise[i]/looks);
    }
}

#endif //DZZ_DISTORT_H
