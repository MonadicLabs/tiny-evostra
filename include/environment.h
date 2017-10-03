#pragma once

#include <tiny_dnn/tiny_dnn.h>

class Environment
{
public:
    Environment(){}
    virtual ~Environment(){}

    // Perform action, return new state and reward
    void step( const tiny_dnn::tensor_t action, tiny_dnn::tensor_t& new_state, double& reward )
    {
        perform_action( action );
        new_state = state();
        reward = reward;
    }

    virtual double reward(){ return 0.0; }
    virtual tiny_dnn::tensor_t state(){ return tiny_dnn::tensor_t(); }
    virtual void perform_action( tiny_dnn::tensor_t action ){}

private:

protected:

};
