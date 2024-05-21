/**
 * @file losses.h
 * @brief This file contains the declarations for loss functions used in neural networks.
 *
 * This file provides the declarations for various loss functions that can be used in neural networks.
 * Loss functions are used to measure the discrepancy between the predicted output and the actual output.
 * These functions are commonly used in training neural networks to optimize the model's performance.
 *
 * @note This file is part of the lightweight-neural-nets library.
 */
#pragma once

/**
 * @brief Enum representing the type of loss function.
 */
typedef enum
{
    LOSS_TYPE_FF,
    LOSS_TYPE_SYMBA
} LossType;

/**
 * @brief Struct representing a Loss function and its partial derivatives.
 */
typedef struct
{
    // Loss type.
    LossType type;
    // Loss function handler.
    double (*loss)(const double, const double, const double);
    // Partial derivative of the loss function for the positive pass.
    double (*pdloss_pos)(const double, const double, const double);
    // Partial derivative of the loss function for the negative pass.
    double (*pdloss_neg)(const double, const double, const double);
} Loss;

/**
 * @brief FF original loss function.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The loss value.
 */
double ff_loss(const double g_pos, const double g_neg, const double threshold);

/**
 * @brief Partial derivative of the FF loss function for the positive pass.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the loss function for the positive pass.
 */
double ff_pdloss_pos(const double g_pos, const double g_neg, const double threshold);

/**
 * @brief Partial derivative of the FF loss function for the negative pass.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the loss function for the negative pass.
 */
double ff_pdloss_neg(const double g_pos, const double g_neg, const double threshold);

/**
 * @brief SymBa loss function.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The loss value.
 */
double symba_loss(const double g_pos, const double g_neg, const double threshold);

/**
 * @brief Partial derivative of the SymBa loss function for the positive pass.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the loss function for the positive pass.
 */
double symba_pdloss_pos(const double g_pos, const double g_neg, const double threshold);

/**
 * @brief Partial derivative of the SymBa loss function for the negative pass.
 *
 * @param g_pos The positive gradient.
 * @param g_neg The negative gradient.
 * @param threshold The threshold value.
 * @return The partial derivative of the loss function for the negative pass.
 */
double symba_pdloss_neg(const double g_pos, const double g_neg, const double threshold);

/**
 * @brief Selects a loss function based on the given type.
 *
 * @param loss_type The type of loss function.
 * @return The selected loss function.
 */
Loss select_loss(LossType loss_type);

/**
 * @brief Original Forward Forward loss function.
 */
#define LOSS_FF \
    ((Loss){LOSS_TYPE_FF, ff_loss, ff_pdloss_pos, ff_pdloss_neg})

/**
 * @brief SymBa loss function.
 */
#define LOSS_SYMBA \
    ((Loss){LOSS_TYPE_SYMBA, symba_loss, symba_pdloss_pos, symba_pdloss_neg})
