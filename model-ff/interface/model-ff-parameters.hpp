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
    float threshold = 5.0;
    float beta1 = 0.9;
    float beta2 = 0.999;
    LossType loss = LossType::LOSS_TYPE_FF;
};

#endif // MODEL_FF_PARAMETERS_HPP