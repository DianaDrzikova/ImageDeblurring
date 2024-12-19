#ifndef REGULARIZERS_H
#define REGULARIZERS_H

#include "stochastic_deconvolution.h"

double regularizer_energy_TV(sd_data *data, int x, int y);
double regularizer_energy_gamma(sd_data *data, int x, int y);
double regularizer_energy_combination(sd_data *data, int x, int y);
double regularizer_energy_sparse_1st2nd(sd_data *data, int x, int y);
double regularizer_energy_data_dependent(sd_data *data, int x, int y);
double regularizer_energy(sd_data *data, int x, int y);

#endif // REGULARIZERS_H