#pragma once



typedef struct
{
    // Loss function.
    double (*loss)(const double, const double, const double);
    // Partial derivative of the loss function for the positive pass.
    double (*pdloss_pos)(const double, const double, const double);
    // Partial derivative of the loss function for the negative pass.
    double (*pdloss_neg)(const double, const double, const double);
} Loss;


// FF original loss function.
double ff_loss(const double g_pos, const double g_neg, const double threshold);
double ff_pdloss_pos(const double g_pos, const double g_neg, const double threshold);
double ff_pdloss_neg(const double g_pos, const double g_neg, const double threshold);

// SymBa loss function.
double symba_loss(const double g_pos, const double g_neg, const double threshold);
double symba_pdloss_pos(const double g_pos, const double g_neg, const double threshold);
double symba_pdloss_neg(const double g_pos, const double g_neg, const double threshold);

// Macros for on-demand Loss struct instantiation
#define LOSS_FF (Loss){ff_loss, ff_pdloss_pos, ff_pdloss_neg}
#define LOSS_SYMBA (Loss){symba_loss, symba_pdloss_pos, symba_pdloss_neg}