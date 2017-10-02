#pragma once

#include <tiny_dnn/tiny_dnn.h>

class Environment
{
public:
    Environment(){}
    virtual ~Environment(){}

    // Perform action, return new state and reward
    double step( const tiny_dnn::Tensor& action, tiny_dnn::Tensor& new_state )
    {
        perform_action( action );
        new_state = state();
        return reward;
    }

    virtual double reward(){ return 0.0; }
    virtual tiny_dnn::Tensor state(){ return tiny_dnn::Tensor(); }
    virtual void perform_action( tiny_dnn::Tensor& action ){}

private:

protected:

};
