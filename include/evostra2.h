#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <random>

#include "utils.h"

// #define DEBUG

template< class FloatType = double >
class EvolutionStrategy
{
public:
    EvolutionStrategy( std::vector< FloatType > weights,
                       std::function<FloatType ( std::vector<FloatType> ) > reward_function,
                       size_t populationSize,
                       FloatType sigma = 0.1,
                       FloatType learningRate = 0.001 )
        :_weights(weights), _get_reward_func( reward_function ), _populationSize(populationSize), _sigma(sigma), _learningRate(learningRate)
    {

    }

    virtual ~EvolutionStrategy()
    {

    }

    std::vector< FloatType > getWeights()
    {
        return _weights;
    }

    void run( size_t iterations, size_t print_step = 10 )
    {
        for( int i = 0; i < iterations; ++i )
        {
            if( i % print_step == 0 )
            {
                std::cerr << "iter " << i << " reward=" << _get_reward_func( _weights ) << std::endl;
            }

            std::vector< std::vector< FloatType > > population;
            std::vector< FloatType > rewards( _populationSize );

            for( int j = 0; j < _populationSize; ++j )
            {
                std::vector< FloatType > x;
                randn( x, _weights.size() );
#ifdef DEBUG
                std::cerr << "x" <<  x << std::endl;
#endif
                population.push_back( x );
            }

            for( int j = 0; j < _populationSize; ++j )
            {
                std::vector< FloatType > weights_try = getWeightsTry( _weights, population[j] );
#ifdef DEBUG
                std::cerr << "weights_try " << weights_try << std::endl;
#endif
                rewards[j] = _get_reward_func( weights_try );
#ifdef DEBUG
                // std::cerr << "lol REWARD = " << rewards[j] << std::endl;
#endif
            }

            FloatType meanReward = mean(rewards);
            FloatType stdReward = stddev(rewards);
            subscalar( rewards, meanReward );
            divscalar( rewards, stdReward );

#ifdef DEBUG
            for( int j = 0; j < rewards.size(); ++j )
            {
                std::cerr << "rewards[" << j << "] = " << rewards[j] << std::endl;
            }
#endif

            std::vector< FloatType > fi_epsi( _weights.size() );
            std::fill( fi_epsi.begin(), fi_epsi.end(), 0 );
            for( int j = 0; j < _populationSize; ++j )
            {
                mulscalar( population[j], rewards[j] );
                fi_epsi = add( fi_epsi, population[j] );
            }
            FloatType lFactor = _learningRate / ( (FloatType)_populationSize * _sigma );
            mulscalar( fi_epsi, lFactor );
            _weights = add( _weights, fi_epsi );
        }

        std::cerr << "solution: " << _weights << std::endl;
    }

private:
    std::function<FloatType ( std::vector<FloatType> ) > _get_reward_func;
    FloatType _sigma;
    FloatType _learningRate;
    size_t _populationSize;
    std::vector< FloatType > _weights;

    std::vector< FloatType > getWeightsTry( std::vector<FloatType> w,
                                            std::vector<FloatType> p )
    {
        std::vector< FloatType > ret = w;
        std::vector< FloatType > jittered = p;
        // std::cerr << "SIGMA=" << _sigma << std::endl;
        mulscalar( jittered, _sigma );
        ret = add( w, jittered );
#ifdef DEBUG
        std::cerr << "jittered = " << ret << std::endl;
#endif
        return ret;
    }

};
