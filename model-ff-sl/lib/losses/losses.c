/**
 * @file losses.c
 * @brief This file contains the implementation of various loss functions used in neural networks.
 */

#include <losses/losses.h>
#include <math.h>

// Static function declarations.
static double stable_sigmoid(double x);
static double softplus(double x, double betha);

/**
 * @brief Calculates the feedforward loss for a given positive and negative gradient and a threshold.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The feedforward loss.
 */
double ff_loss(const double g_pos, const double g_neg, const double threshold)
{
    return softplus(-g_pos + threshold, 1.0) + softplus(g_neg - threshold, 1.0);
}

/**
 * @brief Calculates the partial derivative of the feedforward loss with respect to the positive gradient.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the feedforward loss with respect to the positive gradient.
 */
double ff_pdloss_pos(const double g_pos, const double g_neg, const double threshold)
{
    return -stable_sigmoid(threshold - g_pos);
}

/**
 * @brief Calculates the partial derivative of the feedforward loss with respect to the negative gradient.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the feedforward loss with respect to the negative gradient.
 */
double ff_pdloss_neg(const double g_pos, const double g_neg, const double threshold)
{
    return stable_sigmoid(g_neg - threshold);
}

/**
 * @brief Calculates the stable sigmoid function for a given input.
 *
 * @param x The input value.
 * @return The sigmoid value.
 */
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

/**
 * @brief Calculates the softplus function for a given input and beta value.
 *
 * @param x The input value.
 * @param betha The beta value.
 * @return The softplus value.
 */
static double softplus(double x, double betha)
{
    return log(1.0 + exp(x * betha)) / betha;
}

/**
 * @brief Calculates the SymBa loss for a given positive and negative gradient and a threshold.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The SymBa loss.
 */
double symba_loss(const double g_pos, const double g_neg, const double threshold)
{
    return softplus(threshold * (g_pos - g_neg), 0.1);
}

/**
 * @brief Calculates the partial derivative of the SymBa loss with respect to the positive gradient.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the SymBa loss with respect to the positive gradient.
 */
double symba_pdloss_pos(const double g_pos, const double g_neg, const double threshold)
{
    return threshold * stable_sigmoid(-threshold * (g_pos - g_neg));
}

/**
 * @brief Calculates the partial derivative of the SymBa loss with respect to the negative gradient.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the SymBa loss with respect to the negative gradient.
 */
double symba_pdloss_neg(const double g_pos, const double g_neg, const double threshold)
{
    return -threshold * stable_sigmoid(-threshold * (g_pos - g_neg));
}