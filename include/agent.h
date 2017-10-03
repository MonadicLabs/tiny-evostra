#pragma once

#include "instance.h"

#include <tiny_dnn/tiny_dnn.h>

class Agent : public EvoStraInstance
{
public:
    Agent(){}
    virtual ~Agent(){}

    // Return action tensor, given state
    virtual tiny_dnn::tensor_t step( tiny_dnn::tensor_t state ){ return tiny_dnn::tensor_t(); }

private:

protected:

};
