#include <iostream>

#include <unistd.h>

#include <tiny_dnn/tiny_dnn.h>

#include "include/gym/gym.h"

#include "evostra2.h"
#include "utils.h"

// #define DEBUG
using namespace std;
using namespace tiny_dnn;

std::vector<double> solution = { 0.5, 0.2, 0.3 };
std::vector<double> nn_input = { 1,2,3,4,5 };

network< sequential > nn;
boost::shared_ptr<Gym::Client> client;
boost::shared_ptr<Gym::Environment> env;

class Agent
{
public:

private:

protected:

};

double get_reward( std::vector<double> w )
{
    setParameters(nn, w);
    vec_t res = nn.predict( nn_input );
    std::vector<double> resd;
    for( auto vv : res )
        resd.push_back( vv );
    return -sqnorm( resd, solution );
}

static
void run_single_environment(
        const boost::shared_ptr<Gym::Client>& client,
        const std::string& env_id,
        int episodes_to_run)
{

    boost::shared_ptr<Gym::Space> action_space = env->action_space();
    boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

    for (int e=0; e<episodes_to_run; ++e) {
        printf("%s episode %i...\n", env_id.c_str(), e);
        Gym::State s;
        env->reset(&s);
        float total_reward = 0;
        int total_steps = 0;
        while (1) {
            std::vector<float> action = action_space->sample();
            try{
            env->step(action, true, &s);
            assert(s.observation.size()==observation_space->sample().size());
            total_reward += s.reward;
            total_steps += 1;
            }
            catch( ... )
            {
                cerr << "GYM ERROR ! " << endl;
            }

            if (s.done) break;
        }
        printf("%s episode %i finished in %i steps with reward %0.2f\n",
               env_id.c_str(), e, total_steps, total_reward);
    }
}

double get_reward2( std::vector<double> w )
{
    double total_reward = 0.0;
    setParameters( nn, w );

    int num_episodes = 1;
    Gym::State s;

    for( int episode = 0; episode < num_episodes; ++episode )
    {

        boost::shared_ptr<Gym::Space> action_space = env->action_space();
        boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

        std::vector<float> action = action_space->sample();
        std::vector<float> state = observation_space->sample();
        /*
        cerr << "action_size" << action.size() << endl;
        cerr << "state_size" << state.size() << endl;
        */
        env->reset(&s);

        while(true)
        {
            state = s.observation;
            vec_t nn_state( state.size() );
            std::copy( state.begin(), state.end(), nn_state.begin() );

            vec_t nn_action = nn.predict( nn_state );
            std::copy( nn_action.begin(), nn_action.end(), action.begin() );

            env->step(action, true, &s);
            assert(s.observation.size()==observation_space->sample().size());
            total_reward += s.reward;
            if (s.done) break;
        }
    }

    return total_reward;

}

int main( int argc, char** argv )
{

    /*
    try {
        boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
        run_single_environment(client, "BipedalWalker-v2", 30);

    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    */

    srand( time(NULL) );

    client = Gym::client_create("127.0.0.1", 5000);
    env = client->make("CartPole-v0");

    nn << fully_connected_layer<tiny_dnn::activation::relu>( 4, 8 )
       << fully_connected_layer<relu>( 8 ,16 )
       << fully_connected_layer<activation::tan_h>( 16,1 );
    nn.init_weight();

    std::vector<double> weights = getParameters( nn );
    // randn( weights, 3 );

    EvolutionStrategy<> * es = new EvolutionStrategy<>( weights, get_reward2, 20, 0.1, 0.05 );
    es->run( 3000, 1 );

    setParameters( nn, es->getWeights() );
    vec_t res = nn.predict( nn_input );
    std::vector<double> resd;
    for( auto vv : res )
        resd.push_back( vv );

    std::cerr << "RESULTS: " << resd << endl;

    return 0;
}
