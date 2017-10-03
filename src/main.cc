#include <iostream>

#include <unistd.h>

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
        init();
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

    virtual tensor_t getParameters()
    {
        tensor_t ret(1);
        for( auto popo = _nn.begin(); popo != _nn.end(); popo++ )
        {
            // cerr << (*popo)->weights().size() << " - " << (*popo)->weights()[0]->size() << " - " << (*popo)->weights()[1]->size() << endl;
            for( int k = 0; k < (*popo)->weights().size(); ++k )
            {
                for( int i = 0; i < (*popo)->weights()[k]->size(); ++i )
                {
                    ret[0].push_back( (*popo)->weights()[k]->operator[](i) );
                }
            }
        }
        return ret;
    }

    virtual void setParameters( tensor_t& params )
    {
        int offset = 0;
        for( auto popo = _nn.begin(); popo != _nn.end(); popo++ )
        {
            for( int k = 0; k < (*popo)->weights().size(); ++k )
            {
                for( int i = 0; i < (*popo)->weights()[k]->size(); ++i )
                {
                    (*popo)->weights()[k]->operator[](i) = params[0][offset];
                    offset++;
                }
            }
        }

    }

private:
    void init()
    {
        _nn << fully_connected_layer<tan_h>( 10, 64 )
            << fully_connected_layer<tan_h>( 64, 64 )
            << fully_connected_layer<tan_h>( 64 , 1 );
        _nn.init_weight();
    }

protected:
    network<sequential> _nn;

};

int main( int argc, char** argv )
{
    srand(time(NULL));
    Agent* a = new ExampleAgent();
    tensor_t tinput(1);
    tinput[0] = vec_t(10);
    tensor_t toutput = a->getParameters();
    toutput[0][ toutput[0].size() - 1 ] = 777;
    a->setParameters( toutput );
    for( int k = 0; k < toutput[0].size(); ++k )
    {
        cerr << "t[" << k << "]=" << toutput[0][k] << endl;
    }
    toutput = a->step(tinput);
    for( int k = 0; k < toutput[0].size(); ++k )
    {
        cerr << "x[" << k << "]=" << toutput[0][k] << endl;
    }
    return 0;
}
