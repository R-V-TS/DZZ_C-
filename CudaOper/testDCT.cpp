//
// Created by k504-r on 02.09.19.
//

#include "cuda_heder.h"
#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>

int main(int args, char **argv) {
    /*if(args != 3)
    {
        std::cout << "Enter the image name!" << "\n";
        return -2;
    }*/

    cv::Mat image;
    //image = cv::imread(argv[1], 1);
    image = cv::imread("../imset/Zoneplate.png", 1);

    if(!image.data) {
        return -1;
    }

    std::cout << "Image cols: " << image.cols << "\nImage rows: " << image.rows << "\nImage channels: " << image.channels() << "\n";

    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image", image);
    cv::waitKey(0);

    char *pixel_ride = new char[image.cols*image.rows*image.channels()];
    int pixel_flag = 0;
    cv::Mat Image_gray;
    cv::Mat A(image.size(), image.type());

    double time = 0.0;
    double SMD_image = 0;
    uchar *data = image.data;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols*image.channels(); j++) {
            *(pixel_ride + pixel_flag) = (char) *(data + (image.cols*i*image.channels()) +  j);
            pixel_flag++;
        }
    }

    unsigned int start_time = clock();
    char *pixel_result = DCT_Filrer(pixel_ride, image.cols*image.channels(), image.rows, image.channels());
    unsigned int finish_time = clock();
    std::cout << "Time monothread = " << (float) (finish_time - start_time) / CLOCKS_PER_SEC << " s\n";
    time += (float) (finish_time - start_time) / CLOCKS_PER_SEC;

    pixel_flag = 0;
    for (int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols*image.channels(); j+=image.channels())
        {
            A.data[A.channels() * (A.cols * i) + j] = pixel_result[pixel_flag];
            A.data[A.channels() * (A.cols * i) + j + 1] = pixel_result[pixel_flag + 1];
            A.data[A.channels() * (A.cols * i) + j + 2] = pixel_result[pixel_flag + 2];
            pixel_flag+=image.channels();
        }

    cv::imwrite("../imset/result.png", A);
    cv::namedWindow("filtered", cv::WINDOW_AUTOSIZE);
    cv::imshow("filtered", A);
    cv::waitKey(0);

    std::cout << "All processing time = " << time << " s\n";

    return 0;
}