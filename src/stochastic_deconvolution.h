#ifndef STOCHASTIC_DECONVOLUTION_H
#define STOCHASTIC_DECONVOLUTION_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct sd_data {
    Mat input;      // blurred input image q
    Mat intrinsic;  // intrinsic image p
    Mat blurred;    // A p
    Mat mask;       // mask for valid pixels (1.0 = valid, 0.0 = invalid/padded/saturated)
    string selected_regularizer;
};

struct sd_sample {
    int x, y;
    double ed;
};

struct sd_callbacks {
    void (*splat)(sd_data *data, sd_sample *x, double weight);
    void (*copy)(sd_data *data, sd_sample *x, sd_sample *y);
    double (*evaluate)(sd_data *data, sd_sample *x);
    void (*mutate)(sd_data *data, sd_sample *x, sd_sample *y);
};

///////////////////////////////////////////////////////////////////////////////
// Regularizer and PSF setup //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
const int    psf_cnt = 9;
const double psf_v[] = { 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0 };
const int    psf_x[] = { -4, -3, -2, -1, 0, 1, 2, 3, 4 };
const int    psf_y[] = { -4, -3, -2, -1, 0, 1, 2, 3, 4 };

const int    reg_cnt = 3;
const int    reg_x[] = {  0,  1,  0 };
const int    reg_y[] = {  0,  0,  1 };

void splat(sd_data *data, sd_sample *x, double weight);
void copy_sample(sd_data *data, sd_sample *x, sd_sample *y);
double evaluate(sd_data *data, sd_sample *x);
void mutate(sd_data *data, sd_sample *x, sd_sample *y);
double stochastic_deconvolution(sd_data *data, sd_callbacks *cb, double ed, int n_mutations);
void sample_normal(double &X, double &Y);
bool inside_image(sd_data *data, int x, int y);
double data_energy(sd_data *data, int x, int y);

#endif // STOCHASTIC_DECONVOLUTION_H