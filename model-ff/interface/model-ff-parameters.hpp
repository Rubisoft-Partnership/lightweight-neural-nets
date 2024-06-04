#ifndef MODEL_FF_PARAMETERS_HPP
#define MODEL_FF_PARAMETERS_HPP

#include <vector>

extern "C"
{
    #include <losses/losses.h>
}

struct ModelFFParameters
{
    std::vector<int> units;
    float threshold;
    float beta1, beta2;
    LossType loss;
};

#endif // MODEL_FF_PARAMETERS_HPP