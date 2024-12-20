#include "stochastic_deconvolution.h"
#include "regularizers.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>


///////////////////////////////////////////////////////////////////////////////
// Method Parameters //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
const double sigma          = 4.0;      // mutation standard deviation
const double reset_prob     = 0.005f;   // russian roulette chain reset probability


// takes a sample and splats its energy into the intrinsic and blurred images
void splat(sd_data *data, sd_sample *x, double weight) {
    Vec3d &p = data->intrinsic.at<Vec3d>(x->y, x->x);
    p[0] += weight * x->ed;  // Red channel
    p[1] += weight * x->ed;  // Green channel
    p[2] += weight * x->ed;  // Blue channel

    for (int i = 0; i < psf_cnt; i++) {
        int tx = x->x + psf_x[i];
        int ty = x->y + psf_y[i];

        if (tx >= 0 && ty >= 0 && tx < data->blurred.cols && ty < data->blurred.rows && data->mask.at<double>(ty,tx) > 0.0) { // && data->mask.at<double>(ty,tx) == 1.0

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
        init += apply_regularizer( data, x->x+reg_x[i], x->y+reg_y[i] );

    // splat positive
    splat( data, x, 1.0 );
    plus_val = data_energy( data, x->x, x->y );
    for( int i=0; i<reg_cnt; i++ )
        plus_val += apply_regularizer( data, x->x+reg_x[i], x->y+reg_y[i] );

    // now negative (remove twice to get negative effect)
    splat( data, x, -2.0 );
    minus_val = data_energy( data, x->x, x->y );
    for( int i=0; i<reg_cnt; i++ )
        minus_val += apply_regularizer( data, x->x+reg_x[i], x->y+reg_y[i] );

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
        if (inside_image(data, tx, ty) && data->mask.at<double>(ty,tx) > 0.0) {
            Vec3d delta = data->blurred.at<Vec3d>(ty, tx) - data->input.at<Vec3d>(ty, tx);
            sum += delta.dot(delta); // Sum squared difference over R, G, B
        }
    }
    return sum;
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