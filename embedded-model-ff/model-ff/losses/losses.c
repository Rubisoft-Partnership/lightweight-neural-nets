/**
 * @file losses.c
 * @brief This file contains the implementation of various loss functions used in neural networks.
 */

#include <losses/losses.h>
#include <math.h>
#include <stdio.h>

// Static function declarations.
static float stable_sigmoid(float x);
static float softplus(float x, float betha);

/**
 * @brief Calculates the feedforward loss for a given positive and negative gradient and a threshold.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The feedforward loss.
 */
float ff_loss(const float g_pos, const float g_neg, const float threshold)
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
float ff_pdloss_pos(const float g_pos, const float g_neg, const float threshold)
{
    (void)g_neg; // Unused parameter
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
float ff_pdloss_neg(const float g_pos, const float g_neg, const float threshold)
{
    (void)g_pos; // Unused parameter
    return stable_sigmoid(g_neg - threshold);
}

/**
 * @brief Calculates the stable sigmoid function for a given input.
 *
 * @param x The input value.
 * @return The sigmoid value.
 */
static float stable_sigmoid(float x)
{
    if (x >= 0)
        return 1.0 / (1.0 + exp(-x) + 1e-8);
    else
    {
        const float exp_x = exp(x);
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
static float softplus(float x, float betha)
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
float symba_loss(const float g_pos, const float g_neg, const float threshold)
{
    return softplus(-threshold * (g_pos - g_neg), 0.05);
}

/**
 * @brief Calculates the partial derivative of the SymBa loss with respect to the positive gradient.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the SymBa loss with respect to the positive gradient.
 */
float symba_pdloss_pos(const float g_pos, const float g_neg, const float threshold)
{
    return -threshold * stable_sigmoid(-threshold * (g_pos - g_neg));
}

/**
 * @brief Calculates the partial derivative of the SymBa loss with respect to the negative gradient.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the SymBa loss with respect to the negative gradient.
 */
float symba_pdloss_neg(const float g_pos, const float g_neg, const float threshold)
{
    return threshold * stable_sigmoid(-threshold * (g_pos - g_neg));
}

/**
 * Selects the loss function based on the given loss type.
 *
 * @param loss_type The type of loss function to select.
 * @return The selected loss function.
 */
Loss select_loss(LossType loss_type)
{
    Loss loss;
    switch (loss_type)
    {
    case LOSS_TYPE_FF:
        loss = LOSS_FF;
        break;
    case LOSS_TYPE_SYMBA:
        loss = LOSS_SYMBA;
        break;
    default:
        printf("Unknown loss function type %d. Setting default loss function.", loss_type);
        loss = LOSS_FF;
    }
    return loss;
}
