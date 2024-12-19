#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "regularizers.h"
#include "stochastic_deconvolution.h"

using namespace std;
using namespace cv;

const int    num_iterations = 100;      // number of 'iterations'
double ed                   = 0.025;    // starting deposition energy

static inline double my_drand48() {
    return double(rand())/double(RAND_MAX);
}
#define drand48 my_drand48
Mat blur_image( const Mat &img );


int main( int argc, char **argv ){
    srand((unsigned)time(NULL)); // seed random number generator

    // load input image and process
    Mat input = imread("dandelion.jpg", IMREAD_COLOR);
    input.convertTo(input, CV_64FC3, 1.0 / 255.0);

    int pad = 4;
    Mat padded_input;
    copyMakeBorder(input, padded_input, pad, pad, pad, pad, BORDER_REFLECT, Scalar(0, 0, 0));
    Mat mask = Mat::zeros(padded_input.size(), CV_64F);

    double saturation_threshold = 0.99;
    for(int y = pad; y < padded_input.rows - pad; y++){
        for(int x = pad; x < padded_input.cols - pad; x++){
            Vec3d val = padded_input.at<Vec3d>(y,x);
            double max_val = max(val[0], val[1]);
            max_val = max(val[2], max_val);

            double min_val = min(val[0], val[1]);
            min_val = min(min_val, val[2]);

            double difference_chroma = max_val - min_val;

            double saturation = 0.0;
            if (max_val > 0.0) { 
                saturation = difference_chroma / max_val; 
            }

            if (saturation <= saturation_threshold)
                mask.at<double>(y, x) = 1.0; // Mark as valid
            else 
                mask.at<double>(y, x) = 0.0; // Mark as invalid
        }
    }

    Mat blurred = blur_image( padded_input );
    Mat solution = blurred.clone();

    sd_data data;
    data.input = blurred;   // q
    data.intrinsic = solution; // p
    data.blurred = blur_image( solution ); // A p
    data.mask = mask;

    sd_callbacks cb;
    cb.copy = copy_sample;
    cb.evaluate = evaluate;
    cb.mutate = mutate;
    cb.splat = splat;

    double accept_rate = 0.0;
    for( int k=0; k<num_iterations; k++ ){
        accept_rate = stochastic_deconvolution( &data, &cb, ed, data.input.cols*data.input.rows );
        std::cout << "iteration " << k+1 << " of " << num_iterations << ", acceptance rate: " << accept_rate << ", ed: " << ed << std::endl;
        if( accept_rate < 0.4 )
            ed *= 0.5f;
    }
    // resize back to the original image size
    Mat final_intrinsic = data.intrinsic(Rect(pad, pad, input.cols, input.rows)).clone();
    Mat final_input = padded_input(Rect(pad, pad, input.cols, input.rows)).clone();
    // Write out images
    // ground_truth.png
    {
        Mat out = final_input.clone();
        cv::threshold(out, out, 0.0, 0.0, THRESH_TOZERO);
        out = out * 255.0;
        Mat out8u; out.convertTo(out8u, CV_8U);
        imwrite("ground_truth.png", out8u);
    }

    // blurred.png
    {
        Mat out = blurred(Rect(pad, pad, input.cols, input.rows)).clone();
        cv::threshold(out, out, 0.0, 0.0, THRESH_TOZERO);
        out = out * 255.0;
        Mat out8u; out.convertTo(out8u, CV_8U);
        imwrite("blurred.png", out8u);
    }

    // intrinsic.png
    {
        Mat out = final_intrinsic.clone();
        // clamp to >=0.0
        cv::threshold(out, out, 0.0, 0.0, THRESH_TOZERO);
        // clamp to <=1.0
        cv::min(out, 1.0, out);
        out = out * 255.0;
        Mat out8u; out.convertTo(out8u, CV_8U);
        imwrite("intrinsic.png", out8u);
    }
    std::cout << "Done!" << std::endl;
    return 0;
}


// blurs the input image using the hardcoded PSF above
Mat blur_image(const Mat &img) {
    Mat blurred = Mat::zeros(img.size(), CV_64FC3);  // Initialize the output image

    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            Vec3d val = img.at<Vec3d>(j, i);  // Get the RGB values at (j, i)
            for (int k = 0; k < psf_cnt; k++) {
                int tx = i + psf_x[k];
                int ty = j + psf_y[k];
                if (tx >= 0 && ty >= 0 && tx < img.cols && ty < img.rows) {
                    Vec3d &b = blurred.at<Vec3d>(ty, tx);
                    b[0] += psf_v[k] * val[0];  // Red channel
                    b[1] += psf_v[k] * val[1];  // Green channel
                    b[2] += psf_v[k] * val[2];  // Blue channel
                }
            }
        }
    }
    return blurred;
}