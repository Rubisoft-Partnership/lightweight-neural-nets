#include <losses/losses.h>

#include <math.h>

static double stable_sigmoid(double x);

double ff_loss(const double g_pos, const double g_neg, const double threshold)
{
    return log(1.0 + exp(-g_pos + threshold)) + log(1.0 + exp(g_neg - threshold));
}

double ff_pdloss_pos(const double g_pos, const double g_neg, const double threshold)
{
    return -stable_sigmoid(threshold - g_pos);
}

double ff_pdloss_neg(const double g_pos, const double g_neg, const double threshold)
{
    return stable_sigmoid(g_neg - threshold);
}

static double stable_sigmoid(double x)
{
    if (x >= 0)
        return 1.0 / (1.0 + exp(-x) + 1e-8);
    else
    {
        const double exp_x = exp(x);
        return exp_x / (1.0 + exp_x + 1e-8);
    }
}

double symba_loss(const double g_pos, const double g_neg, const double threshold)
{
    return log(1.0 + exp(threshold * (g_pos - g_neg)));
}

double symba_pdloss_pos(const double g_pos, const double g_neg, const double threshold)
{
    return -threshold * g_neg * stable_sigmoid(threshold * (g_neg - g_pos));
}

double symba_pdloss_neg(const double g_pos, const double g_neg, const double threshold)
{
    return threshold * g_pos * stable_sigmoid(threshold * (g_neg - g_pos));
}