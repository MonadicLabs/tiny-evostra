
#include "utils.h"

#include <iostream>
#include <random>
#include <numeric>

using namespace std;

void print_tensor( tiny_dnn::tensor_t& t )
{
    for( int i = 0; i < t.size(); ++i )
    {
        cerr << "[ ";
        for( int j = 0; j < t[i].size(); ++j )
        {
            cerr << t[i][j] << " ";
        }
        cerr << " ]" << endl;
    }
}

tiny_dnn::tensor_t randn(int sz, double mean, double stddev)
{
    tiny_dnn::tensor_t ret(1);
    ret[0].resize(sz);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution( mean, stddev );
    for( int k = 0; k < sz; ++k )
    {
        ret[0][k] = distribution(generator);
    }
    return ret;
}

tiny_dnn::tensor_t add_tensor(const tiny_dnn::tensor_t &a, const tiny_dnn::tensor_t &b, double alpha )
{
    tiny_dnn::tensor_t ret;
    for( int i = 0; i < a.size(); ++i )
    {
        ret.push_back( tiny_dnn::vec_t() );
        for( int j = 0; j < a[i].size(); ++j )
        {
            ret[i].push_back( a[i][j] + b[i][j] * alpha );
        }
    }
    return ret;
}

double meanv(const tiny_dnn::vec_t &v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    return mean;
}

double stddev(const tiny_dnn::vec_t &v)
{
    double mean = meanv(v);
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    return stdev;
}
