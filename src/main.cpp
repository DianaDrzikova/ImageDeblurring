#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////
// Data structures and function prototypes ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
struct sd_data {
    Mat input;      // blurred input image q
    Mat intrinsic;  // intrinsic image p
    Mat blurred;    // A p
};

struct sd_sample {
    int x, y;
    double ed;
};

struct sd_callbacks {
    void (*splat)( struct sd_data *data, struct sd_sample *x, double weight );
    void (*copy)( struct sd_data *data, struct sd_sample *x, struct sd_sample *y );
    double (*evaluate)( struct sd_data *data, struct sd_sample *x );
    void (*mutate)( struct sd_data *data, struct sd_sample *x, struct sd_sample *y );
};

///////////////////////////////////////////////////////////////////////////////
// Method Parameters //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
const double reg_weight     = 0.0005;   // regularizer weight
const double sigma          = 4.0;      // mutation standard deviation
const double reset_prob     = 0.005f;   // russian roulette chain reset probability
const int    num_iterations = 100;      // number of 'iterations'
double ed                   = 0.025;    // starting deposition energy

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

///////////////////////////////////////////////////////////////////////////////
// Forward declarations of callbacks and auxiliary functions //////////////////
///////////////////////////////////////////////////////////////////////////////
static inline double my_drand48() {
    return double(rand())/double(RAND_MAX);
}
#define drand48 my_drand48

void splat( sd_data *data, sd_sample *x, double weight );
void copy_sample( sd_data *data, sd_sample *x, sd_sample *y );
double evaluate( sd_data *data, sd_sample *x );
void mutate( sd_data *data, sd_sample *x, sd_sample *y );
double stochastic_deconvolution(sd_data *data, sd_callbacks *cb, double ed, int n_mutations);

Mat load_grayscale( const char *filename );
Mat blur_image( const Mat &img );
void sample_normal( double &X, double &Y );
bool inside_image( sd_data *data, int x, int y );
double data_energy( sd_data *data, int x, int y );
double regularizer_energy( sd_data *data, int x, int y );

///////////////////////////////////////////////////////////////////////////////
// Entry point ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char **argv ){
    srand((unsigned)time(NULL)); // seed random number generator

    // load input image and process
    Mat input = imread("dandelion.jpg", IMREAD_COLOR);
    input.convertTo(input, CV_64FC3, 1.0 / 255.0);
    Mat blurred = blur_image( input );
    Mat solution = blurred.clone();

    sd_data data;
    data.input = blurred;   // q
    data.intrinsic = solution; // p
    data.blurred = blur_image( solution ); // A p

    sd_callbacks cb;
    cb.copy = copy_sample;
    cb.evaluate = evaluate;
    cb.mutate = mutate;
    cb.splat = splat;

    double accept_rate = 0.0;
    for( int k=0; k<num_iterations; k++ ){
        accept_rate = stochastic_deconvolution( &data, &cb, ed, input.cols*input.rows );
        std::cout << "iteration " << k+1 << " of " << num_iterations << ", acceptance rate: " << accept_rate << ", ed: " << ed << std::endl;
        if( accept_rate < 0.4 )
            ed *= 0.5f;
    }

    // Write out images
    // ground_truth.png
    {
        Mat out = input.clone();
        cv::threshold(out, out, 0.0, 0.0, THRESH_TOZERO);
        out = out * 255.0;
        Mat out8u; out.convertTo(out8u, CV_8U);
        imwrite("ground_truth.png", out8u);
    }

    // blurred.png
    {
        Mat out = data.input.clone();
        cv::threshold(out, out, 0.0, 0.0, THRESH_TOZERO);
        out = out * 255.0;
        Mat out8u; out.convertTo(out8u, CV_8U);
        imwrite("blurred.png", out8u);
    }

    // intrinsic.png
    {
        Mat out = data.intrinsic.clone();
        // clamp to >=0.0
        cv::threshold(out, out, 0.0, 0.0, THRESH_TOZERO);
        // clamp to <=1.0
        cv::min(out, 1.0, out);
        out = out * 255.0;
        Mat out8u; out.convertTo(out8u, CV_8U);
        imwrite("intrinsic.png", out8u);
    }

    return 0;
}

// loads an image and converts it to grayscale by averaging and normalizing to [0,1]
Mat load_grayscale( const char *filename ){
    Mat input = imread(filename, IMREAD_COLOR);
    if( input.empty() ) {
        std::cerr << "Could not load image: " << filename << std::endl;
        exit(1);
    }
    input.convertTo(input, CV_64F, 1.0/255.0);

    vector<Mat> ch;
    split(input, ch);
    Mat gray = (ch[0] + ch[1] + ch[2]) / 3.0;

    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);
    gray = (gray - minVal) / (maxVal - minVal);

    return gray;
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

// takes a sample and splats its energy into the intrinsic and blurred images
void splat(sd_data *data, sd_sample *x, double weight) {
    Vec3d &p = data->intrinsic.at<Vec3d>(x->y, x->x);
    p[0] += weight * x->ed;  // Red channel
    p[1] += weight * x->ed;  // Green channel
    p[2] += weight * x->ed;  // Blue channel

    for (int i = 0; i < psf_cnt; i++) {
        int tx = x->x + psf_x[i];
        int ty = x->y + psf_y[i];
        if (tx >= 0 && ty >= 0 && tx < data->blurred.cols && ty < data->blurred.rows) {
            Vec3d &b = data->blurred.at<Vec3d>(ty, tx);
            b[0] += weight * x->ed * psf_v[i];  // Red channel
            b[1] += weight * x->ed * psf_v[i];  // Green channel
            b[2] += weight * x->ed * psf_v[i];  // Blue channel
        }
    }
}

void copy_sample( sd_data *data, sd_sample *x, sd_sample *y ){
    y->x = x->x;
    y->y = x->y;
    y->ed = x->ed;
}

// evaluates the energy change of applying the sample
double evaluate( sd_data *data, sd_sample *x ){
    double init=0.0, plus_val=0.0, minus_val=0.0;
    double de_plus, de_minus;

    // initial energy
    init = data_energy( data, x->x, x->y );
    for( int i=0; i<reg_cnt; i++ )
        init += regularizer_energy( data, x->x+reg_x[i], x->y+reg_y[i] );

    // splat positive
    splat( data, x, 1.0 );
    plus_val = data_energy( data, x->x, x->y );
    for( int i=0; i<reg_cnt; i++ )
        plus_val += regularizer_energy( data, x->x+reg_x[i], x->y+reg_y[i] );

    // now negative (remove twice to get negative effect)
    splat( data, x, -2.0 );
    minus_val = data_energy( data, x->x, x->y );
    for( int i=0; i<reg_cnt; i++ )
        minus_val += regularizer_energy( data, x->x+reg_x[i], x->y+reg_y[i] );

    // restore original
    splat( data, x, 1.0 );

    de_plus  = init - plus_val;
    de_minus = init - minus_val;

    if( de_minus > de_plus ){
        x->ed = -x->ed;
        return de_minus;
    }

    return de_plus;
}

// mutates the sample
void mutate( sd_data *data, sd_sample *x, sd_sample *y ){
    double dx, dy;
    y->ed = x->ed;
    if( drand48() >= reset_prob ){
        while(true){
            sample_normal( dx, dy );
            y->x = int(double(x->x) + sigma*dx + 0.5);
            y->y = int(double(x->y) + sigma*dy + 0.5);
            if( (y->x != x->x || y->y != x->y) && inside_image( data, y->x, y->y ) )
                break;
        }
    } else {
        y->x = int(drand48()*data->intrinsic.cols);
        y->y = int(drand48()*data->intrinsic.rows);
    }
}

double data_energy(sd_data *data, int x, int y) {
    double sum = 0.0;
    for (int i = 0; i < psf_cnt; i++) {
        int tx = x + psf_x[i];
        int ty = y + psf_y[i];
        if (inside_image(data, tx, ty)) {
            Vec3d delta = data->blurred.at<Vec3d>(ty, tx) - data->input.at<Vec3d>(ty, tx);
            sum += delta.dot(delta); // Sum squared difference over R, G, B
        }
    }
    return sum;
}

double regularizer_energy(sd_data *data, int x, int y) {
    if (!inside_image(data, x, y)) return 0.0;

    Vec3d dx = Vec3d(0, 0, 0), dy = Vec3d(0, 0, 0);
    Vec3d val = data->intrinsic.at<Vec3d>(y, x);

    if (x > 0) {
        Vec3d left = data->intrinsic.at<Vec3d>(y, x - 1);
        dx = val - left;
    }
    if (y > 0) {
        Vec3d up = data->intrinsic.at<Vec3d>(y - 1, x);
        dy = val - up;
    }

    return reg_weight * sqrt(dx.dot(dx) + dy.dot(dy)); // Multi-channel gradient magnitude
}

void sample_normal( double &X, double &Y ){
    double u=drand48(), v=drand48();
    double lnu = sqrt( -2.0*log(u) );
    X = lnu*cos(2.0*M_PI*v);
    Y = lnu*sin(2.0*M_PI*v);
}

bool inside_image( sd_data *data, int x, int y ){
    return (x >= 0 && x < data->blurred.cols && y >= 0 && y < data->blurred.rows);
}

double stochastic_deconvolution(sd_data *data, sd_callbacks *cb, double ed, int n_mutations){
    double fx, fy, a_rate = 0.0f;
    sd_sample sample_x, sample_y;

    sample_x.x = int(drand48()*data->input.cols);
    sample_x.y = int(drand48()*data->input.rows);
    sample_x.ed = ed;

    fx = cb->evaluate( data, &sample_x );

    for( int i=0; i<n_mutations; i++ ){
        cb->mutate( data, &sample_x, &sample_y );
        fy = cb->evaluate( data, &sample_y );

        if( fy > 0.0 ){
            a_rate += 1.0f;
            cb->splat( data, &sample_y, 1.0 );
        }

        if( (fx <= 0.0 && fy >= fx) || (drand48() < std::min(1.0, fy/fx)) ){
            cb->copy( data, &sample_y, &sample_x );
            fy = cb->evaluate( data, &sample_x );
            fx = fy;
        }
    }

    return a_rate/double(n_mutations);
}