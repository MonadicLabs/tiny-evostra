#pragma once

#include "agent.h"
#include "environment.h"
#include "utils.h"

#include <memory>
#include <thread>
#include <iostream>

using namespace std;

template< class EnvType >
class Experiment
{
public:
    void start( std::shared_ptr< Agent > agent )
    {
        _agent = agent;
        _environement = std::make_shared<EnvType>();
        _expThread = std::thread( &Experiment::run, this );
        _cumReward = 0.0;
    }

    void run()
    {
        for( int i = 0; i < 10; ++i )
        {
            tiny_dnn::tensor_t s = _environement->state();
#ifdef DEBUG
            cerr << "received state: ";
            print_tensor( s );
#endif
            tiny_dnn::tensor_t a = _agent->step( s );
#ifdef DEBUG
            cerr << "selected action: ";
            print_tensor( a );
#endif
            _environement->perform_action( a );
            double reward = _environement->reward();
#ifdef DEBUG
            cerr << "received reward: " << reward << endl;
#endif
            _cumReward += reward;
        }
    }

    void waitForTermination()
    {
        return _expThread.join();
    }

    double getCumulatedReward()
    {
        return _cumReward;
    }

private:
    std::shared_ptr< Agent > _agent;
    std::shared_ptr< Environment > _environement;
    std::thread _expThread;
    double _cumReward;
};
