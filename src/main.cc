#include <iostream>

#include <tiny_dnn/tiny_dnn.h>

#include "tiny_evostra.h"

using namespace std;
using namespace tiny_dnn;

// Simple example.
// RL agent will try to predict next value of sin( ) from past 10 values
// Reward is inverse of L2 loss

class ExampleAgent : public Agent
{
public:
    ExampleAgent()
    {

    }

    virtual ~ExampleAgent()
    {

    }

    virtual tiny_dnn::tensor_t step( tiny_dnn::tensor_t state )
    {
        return _nn.predict(
                    state
                    );
    }

private:

protected:
    network<sequential> _nn;

};

int main( int argc, char** argv )
{
    return 0;
}
