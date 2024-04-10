#include <losses/losses.h>

#include <math.h>

static double stable_sigmoid(double x);
static double softplus(double x, double betha);

double ff_loss(const double g_pos, const double g_neg, const double threshold)
{
    return softplus(-g_pos + threshold, 1.0) + softplus(g_neg - threshold, 1.0);
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

static double softplus(double x, double betha)
{
    return log(1.0 + exp(x * betha)) / betha;
}

double symba_loss(const double g_pos, const double g_neg, const double threshold)
{
    return softplus(threshold * (g_pos - g_neg), 0.1);
}

double symba_pdloss_pos(const double g_pos, const double g_neg, const double threshold)
{
    return threshold * stable_sigmoid(-threshold * (g_pos - g_neg));
}

double symba_pdloss_neg(const double g_pos, const double g_neg, const double threshold)
{
    return -threshold * stable_sigmoid(-threshold * (g_pos - g_neg));
}