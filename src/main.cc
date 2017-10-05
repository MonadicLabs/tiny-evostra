#include <iostream>

#include <unistd.h>

#include <tiny_dnn/tiny_dnn.h>

#include "tiny_evostra.h"
#include "evostra.h"
#include "utils.h"

// #define DEBUG
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
#ifdef DEBUG
        print_tensor( state );
#endif
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
        _nn << fully_connected_layer<tan_h>( 10, 10 )
            << fully_connected_layer<tan_h>( 10, 4 )
            << fully_connected_layer<tan_h>( 4 , 1 );
        _nn.init_weight();
    }

protected:
    network<sequential> _nn;

};

class ExampleEnvironment : public Environment
{
public:

    ExampleEnvironment()
    {
        // cerr << "env ctor" << endl;
        _t = 0; // ((double)(rand()) / (double)(RAND_MAX)) * 1000.0;
        _dt = 0.01;
    }

    virtual ~ExampleEnvironment()
    {
        // cerr << "env dtor" << endl;
    }

    virtual double reward()
    {
        return _curReward;
    }

    virtual tiny_dnn::tensor_t state()
    {
        update_state();
        return _curState;
    }

    virtual void perform_action( tiny_dnn::tensor_t action )
    {
        // Compute reward
        double aval = action[0][0];
        _curReward = -((aval - _expectedPred)*(aval - _expectedPred));
    }

private:

    void update_state()
    {
        _curState = tiny_dnn::tensor_t(1);
        _curState[0] = tiny_dnn::vec_t(10);
        for( int k = 0; k < 10; ++k )
        {
            _curState[0][k] = sin( _t + (k * _dt) );
        }
        _t += _dt;
        _expectedPred = sin( _t + (9 * _dt) );
    }

    tiny_dnn::tensor_t _curState;
    double _t;
    double _dt;
    double _expectedPred;
    double _curReward;
    tiny_dnn::tensor_t _lastAction;

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

    EvoStra< ExampleAgent, ExampleEnvironment > evostra;
    evostra.train();

    return 0;
}
