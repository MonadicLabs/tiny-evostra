#pragma once

#include <tiny_dnn/tiny_dnn.h>

void print_tensor( tiny_dnn::tensor_t& t );
tiny_dnn::tensor_t add_tensor(const tiny_dnn::tensor_t& a,
                               const tiny_dnn::tensor_t& b , double alpha = 1.0 );

tiny_dnn::tensor_t randn(int sz, double mean, double stddev);
double meanv(const tiny_dnn::vec_t &v);
double stddev(const tiny_dnn::vec_t &v);
