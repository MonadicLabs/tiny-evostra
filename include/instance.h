#pragma

#include <tiny_dnn/tiny_dnn.h>

class EvoStraInstance
{
public:
    EvoStraInstance(){}
    virtual ~EvoStraInstance(){}

    virtual tiny_dnn::tensor_t getParameters(){ return tiny_dnn::tensor_t(); }
    virtual void setParameter( tiny_dnn::tensor_t& params ){}

private:

protected:

};
