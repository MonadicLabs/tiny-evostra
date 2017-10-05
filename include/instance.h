#pragma

#include <tiny_dnn/tiny_dnn.h>

class EvoStraInstance
{
public:
    EvoStraInstance(){}
    virtual ~EvoStraInstance(){}

    virtual tiny_dnn::tensor_t getParameters(){ return tiny_dnn::tensor_t(); }
    virtual void setParameters( tiny_dnn::tensor_t& params ){}
    virtual void train( std::vector<double> deltaVec )
    {
        tiny_dnn::tensor_t curParams = getParameters();
        for( int k = 0; k < curParams[0].size(); ++k )
        {
            curParams[0][k] += deltaVec[k];
        }
        setParameters( curParams );
    }

private:

protected:

};
