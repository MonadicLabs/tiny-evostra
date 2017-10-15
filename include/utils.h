#pragma once

#include <vector>
#include <random>
#include <iterator>     // std::ostream_iterator
#include <iostream>

#include <unistd.h>
#include <stdlib.h>

#include <tiny_dnn/tiny_dnn.h>

template< class FloatType = double >
FloatType sqnorm( const std::vector<FloatType>& a,
                const std::vector<FloatType>& b )
{
    FloatType r = (FloatType)0;
    for( int i = 0; i < a.size(); ++i )
    {
        r += (a[i]-b[i] )* (a[i]-b[i]);
    }
    return r;
}

template< class FloatType = double >
void randn( std::vector<FloatType>& v, size_t n )
{
    std::default_random_engine _generator;
    _generator.seed( time(NULL) );
    std::normal_distribution<FloatType> distribution(1.0);

    v.resize( n );
    for (int i=0;i<n;i++)
    {
        v.at(i) = (double)rand() / (double)RAND_MAX; // distribution(_generator);
    }

}

template< class FloatType = double >
void mulscalar( std::vector<FloatType>& v, FloatType s )
{
    for( FloatType& vv : v )
    {
        // std::cerr << "vv: " << vv << std::endl;
        // std::cerr << "sigma: " << s << std::endl;
        vv *= s;
        // std::cerr << "~vv: " << vv << std::endl;
    }
}

template< class FloatType = double >
void divscalar( std::vector<FloatType>& v, FloatType s )
{
    return mulscalar( v, (FloatType)(1.0 / s) );
}

template< class FloatType = double >
void subscalar( std::vector<FloatType>& v, FloatType s )
{
    for( FloatType& vv : v )
    {
        vv -= s;
    }
}

template< class FloatType = double >
FloatType mean( const std::vector< FloatType >& v )
{
    FloatType ret = (FloatType)0;
    for( FloatType vv : v )
    {
        ret += vv;
    }
    return ret / (FloatType)v.size();
}

template< class FloatType = double >
FloatType stddev( const std::vector< FloatType >& v )
{
    FloatType mv = mean( v );
    FloatType ret = (FloatType)0;
    for( FloatType vv : v )
    {
        ret += (vv - mv)*(vv - mv);
    }
    return ret / (FloatType)v.size();
}

template< class FloatType = double >
std::vector< FloatType > add( const std::vector<FloatType>& a,
                              const std::vector<FloatType>& b )
{
    std::vector<FloatType> ret( a.size() );
    for( int i = 0; i < a.size(); ++i )
    {
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

std::vector<double> getParameters( const tiny_dnn::network<tiny_dnn::sequential>& nn )
{
    std::vector<double> ret;
    for( auto popo = nn.begin(); popo != nn.end(); popo++ )
    {
        for( int k = 0; k < (*popo)->weights().size(); ++k )
        {
            for( int i = 0; i < (*popo)->weights()[k]->size(); ++i )
            {
                ret.push_back( (*popo)->weights()[k]->operator[](i) );
            }
        }
    }
    return ret;
}

void setParameters( tiny_dnn::network<tiny_dnn::sequential>& nn, const std::vector<double>& params )
{
    int offset = 0;
    for( auto popo = nn.begin(); popo != nn.end(); popo++ )
    {
        for( int k = 0; k < (*popo)->weights().size(); ++k )
        {
            for( int i = 0; i < (*popo)->weights()[k]->size(); ++i )
            {
                (*popo)->weights()[k]->operator[](i) = params[offset];
                offset++;
            }
        }
    }
}
