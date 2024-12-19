#include "regularizers.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>


const double reg_weight     = 0.0005;   // regularizer weight
const double gamma_value = 2.0; // gamma value according to paper
const double blend_factor = 0.5; // blend factor to combine gamma and linear reg.

static const double frac_exp = 0.8; // value from Levin paper
static const double epsilon = 1e-3; // robust penalty

double apply_regularizer(sd_data *data, int x, int y){
    if (data->selected_regularizer == "tv") {
        return regularizer_energy_TV(data, x, y);
    } else if (data->selected_regularizer == "gamma") {
        return regularizer_energy_gamma(data, x, y);
    } else if (data->selected_regularizer == "combination") {
        return regularizer_energy_combination(data, x, y);
    } else if (data->selected_regularizer == "sparse") {
        return regularizer_energy_sparse_1st2nd(data, x, y);
    } else if (data->selected_regularizer == "data") {
        return regularizer_energy_data_dependent(data, x, y);
    } else if (data->selected_regularizer == "discontinuous") {
        return regularizer_energy_discontinuous(data, x, y);
    } else {
        cerr << "Invalid regularizer type: " << data->selected_regularizer << endl;
        return 0.0;
    }
}

static inline Vec3d get_pixel_safe(const Mat &img, int x, int y) {
    if(x < 0 || x >= img.cols || y < 0 || y >= img.rows) {
        return Vec3d(0,0,0);
    }
    return img.at<Vec3d>(y,x);
}

// 1) Sparse 1st and 2nd Order Derivatives Regularizer
double regularizer_energy_sparse_1st2nd (sd_data *data, int x, int y) {
    if(!inside_image(data, x, y)) return 0.0;

    Vec3d p = get_pixel_safe(data->intrinsic, x, y);
    Vec3d pxm = get_pixel_safe(data->intrinsic, x-1, y);
    Vec3d pym = get_pixel_safe(data->intrinsic, x, y-1);
    Vec3d pxp = get_pixel_safe(data->intrinsic, x+1, y);
    Vec3d pyp = get_pixel_safe(data->intrinsic, x, y+1);

    // first order differences
    Vec3d dx = p - pxm;    
    Vec3d dy = p - pym;  

    // second order differences (Laplacian components)
    Vec3d dxx = pxp + pxm - 2.0*p;
    Vec3d dyy = pyp + pym - 2.0*p;

    auto frac_norm = [&](double v) { return pow(fabs(v), frac_exp); };

    double cost = 0.0;
    for (int c = 0; c < 3; c++) {
        cost += frac_norm(dx[c]) + frac_norm(dy[c]);
        cost += frac_norm(dxx[c]) + frac_norm(dyy[c]);
    }

    return reg_weight * cost;
}


// 2) Data-Dependent Regularizer
// Uses the gradient of the input image q to modulate the smoothing
double regularizer_energy_data_dependent (sd_data *data, int x, int y) {
    if(!inside_image(data, x, y)) return 0.0;

    Vec3d q = get_pixel_safe(data->input, x, y);
    Vec3d qxm = get_pixel_safe(data->input, x-1, y);
    Vec3d qym = get_pixel_safe(data->input, x, y-1);

    Vec3d dqx = q - qxm;
    Vec3d dqy = q - qym;

    double grad_mag_input = sqrt(dqx.dot(dqx) + dqy.dot(dqy));

    double beta = 10.0;
    double w = exp(-beta * grad_mag_input);

    // Compute first order differences of p
    Vec3d p = get_pixel_safe(data->intrinsic, x, y);
    Vec3d pxm = get_pixel_safe(data->intrinsic, x-1, y);
    Vec3d pym = get_pixel_safe(data->intrinsic, x, y-1);

    Vec3d dx = p - pxm;
    Vec3d dy = p - pym;

    // Weighted L1 norm
    double cost = 0.0;
    for(int c=0; c<3; c++){
        cost += w * fabs(dx[c]) + w * fabs(dy[c]);
    }

    return reg_weight * cost;
}

// 3) Discontinuous (Robust) Regularizer with Heavy-tailed distribution
// Using a Charbonnier-like penalty with fractional exponent:
double regularizer_energy_discontinuous(sd_data *data, int x, int y) {
    if(!inside_image(data, x, y)) return 0.0;

    Vec3d p = get_pixel_safe(data->intrinsic, x, y);
    Vec3d pxm = get_pixel_safe(data->intrinsic, x-1, y);
    Vec3d pym = get_pixel_safe(data->intrinsic, x, y-1);

    Vec3d dx = p - pxm;
    Vec3d dy = p - pym;

    // Charbonnier-like penalty: (d^2 + epsilon^2)^(alpha)
    double alpha = 0.4;

    auto robust_penalty = [&](double d) {
        return pow(d*d + epsilon*epsilon, alpha);
    };

    double cost = 0.0;
    for (int c = 0; c < 3; c++) {
        cost += robust_penalty(dx[c]) + robust_penalty(dy[c]);
    }

    return reg_weight * cost;
}

double regularizer_energy_TV(sd_data *data, int x, int y) {
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
double regularizer_energy_gamma(sd_data *data, int x, int y) {
    if (x < 1 || x >= data->intrinsic.cols - 1 || y < 1 || y >= data->intrinsic.rows - 1)
        return 0.0;

    Vec3d center_val = data->intrinsic.at<Vec3d>(y, x);
    Vec3d center_gamma;
    for (int c = 0; c < 3; c++) {
        center_gamma[c] = pow(center_val[c], 1.0 / gamma_value);
    }

    double sum_abs_diff = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (inside_image(data, nx, ny)) {
                Vec3d neighbor_val = data->intrinsic.at<Vec3d>(ny, nx);
                Vec3d neighbor_gamma;
                for (int c = 0; c < 3; c++) {
                    neighbor_gamma[c] = pow(neighbor_val[c], 1.0 / gamma_value);
                    double diff = fabs(neighbor_gamma[c] - center_gamma[c]);
                    sum_abs_diff += diff;
                }
            }
        }
    }

    return reg_weight * sum_abs_diff;
}

double regularizer_energy_combination(sd_data *data, int x, int y) {
    double lin = regularizer_energy_TV(data, x, y);
    double gamma_sad = regularizer_energy_gamma(data, x, y);
    return blend_factor * lin + (1.0 - blend_factor) * gamma_sad;
}