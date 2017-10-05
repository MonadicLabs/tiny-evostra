#pragma once

#include "agent.h"
#include "environment.h"
#include "experiment.h"
#include "utils.h"

#include <math.h>

#include <memory>
#include <vector>
#include <iostream>

using namespace std;

template< class AgentType, class EnvironmentType >
class EvoStra
{
public:

    void test( std::shared_ptr< Agent > test_agent )
    {
        EnvironmentType env;
        tiny_dnn::tensor_t s = env.state();
        tiny_dnn::tensor_t a = test_agent->step( s );
        env.perform_action( a );
        tiny_dnn::tensor_t sprime = env.state();
        cerr << "s=";
        print_tensor( s );
        cerr << "action=";
        print_tensor( a );
        cerr << "sprime=";
        print_tensor( sprime );
    }

    void train()
    {
        const int populationSize = 50;
        const int numEpochs = 100;
        const double sigma = 0.01;
        const double alpha = 0.0001;

        std::vector< std::shared_ptr< Agent > > _agents( populationSize );
        std::vector< tiny_dnn::tensor_t > _epsilons( populationSize );
        std::vector< tiny_dnn::tensor_t > _thetas( populationSize );
        std::vector< std::shared_ptr< Experiment< EnvironmentType > > > _exps;
        tiny_dnn::vec_t _rewards( populationSize );

        for( int k = 0; k < populationSize; ++k )
        {
            _agents[k] = std::make_shared<AgentType>();
            _epsilons[k] = tiny_dnn::tensor_t(1);
        }

        const int params_size = _agents[0]->getParameters()[0].size();

        // FOR EACH EPOCH
        for( int i = 0; i < numEpochs; ++i )
        {
            cerr << "EPOCH #" << i << endl;

            // Save current parameters
            for( int k = 0; k < populationSize; ++k )
            {
                std::shared_ptr<Agent> a = _agents[k];
                _thetas[k] = a->getParameters();
            }

            // Sample perturbations
            for( int k = 0; k < populationSize; ++k )
            {
                _epsilons[k] = randn( params_size, 0.0, sigma );
            }

            // Add perturbation to net parameters
            for( int k = 0; k < populationSize; ++k )
            {
                tiny_dnn::tensor_t theta = _thetas[k];
                tiny_dnn::tensor_t epsilon = _epsilons[k];
                theta = add_tensor(theta, epsilon );
                _agents[k]->setParameters( theta );
            }

            // START EPISODE
            for( int k = 0; k < populationSize; ++k )
            {
                std::shared_ptr< Experiment< EnvironmentType > > _exp = std::make_shared< Experiment< EnvironmentType > >();
                _exps.push_back( _exp );
                _exp->start( _agents[k] );
            }

            // END OF EPISODE
            for( int k = 0; k < populationSize; ++k )
            {
                std::shared_ptr< Experiment< EnvironmentType > > _exp = _exps[k];
                _exp->waitForTermination();
                // cerr << "accumulated reward: " << _exp->getCumulatedReward() << endl;
                _rewards[k] = _exp->getCumulatedReward();
            }

            // Normalize rewards
            double meanReward = meanv(_rewards);
            double stddevReward = stddev( _rewards );
            double maxReward = -1 * numeric_limits<double>::max();
            double maxRawReward = maxReward;
            int best_idx = -1;
            int ii = 0;
            for( float rv : _rewards )
            {
                if( rv > maxRawReward )
                    maxRawReward = rv;

                rv = ( rv - meanReward ) / stddevReward;

                if( rv > maxReward )
                {
                    maxReward = rv;
                    best_idx = ii;
                }
                ++ii;
            }
            cerr << "best reward: " << maxReward << endl;
            cerr << "best raw reward: " << maxRawReward << endl;

            // Update parameters
            std::vector< double > s_fi_epsi( params_size );

            // Set to zero
            std::fill(s_fi_epsi.begin(), s_fi_epsi.end(), 0.0);
            for( int k = 0; k < populationSize; ++k )
            {
                // Compute sum( Fi * eps_i )
                for( int i = 0; i < params_size; ++i )
                {
                    s_fi_epsi[i] += _rewards[k] * _epsilons[k][0][i];
                }
            }

            // Apply learning rate
            for( int i = 0; i < params_size; ++i )
            {
                s_fi_epsi[i] *= -(1.0 / (((double)populationSize) * sigma)) * alpha;
            }

            // Update parameters for real
            for( int k = 0; k < populationSize; ++k )
            {
                std::shared_ptr<Agent> a = _agents[ k ];
                a->setParameters( _thetas[k] );
                a->train( s_fi_epsi );
            }

            // Reset current experiments
            _exps.clear();

            // TEST PROCEDURE
            test( _agents[best_idx] );
            //
        }
    }

private:

protected:

};
