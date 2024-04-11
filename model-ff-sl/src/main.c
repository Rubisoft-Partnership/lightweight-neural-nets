#include <time.h>
#include <stdlib.h>

#include <ff-net/ff-net.h>
#include <data/data.h>
#include <logging/logging.h>
#include <utils/utils.h>
#include <losses/losses.h>

#include <confusion-matrix/confusion-matrix.h>

// Input and output size is harded coded here as machine learning
// repositories usually don't include the input and output size in the data itself.
const int nips = DATA_FEATURES;
const int nops = DATA_CLASSES;
const int layers_sizes[] = {DATA_FEATURES, 500};

const int layers_number = sizeof(layers_sizes) / sizeof(layers_sizes[0]);
// Hyper Parameters.
double rate = 0.001;
const int epochs = 5;
const double threshold = 4.0;

Data data;
FFNet ffnet;

static void setup(void)
{
    // Comment to repeat the same results
    // set_seed(time(NULL));
    set_log_level(LOG_DEBUG);
    open_log_file_with_timestamp();

    data = build();
    Loss loss_suite = LOSS_FF;
    ffnet = new_ff_net(layers_sizes, layers_number, relu, pdrelu, threshold, loss_suite);
    log_debug("FFNet built with the following layers:");
    increase_indent();
    for (int i = 0; i < ffnet.num_cells; i++)
        log_debug("Layer %d: %d inputs, %d outputs", i, ffnet.layers[i].nips, ffnet.layers[i].nops);
    decrease_indent();
}

static void train_loop(void)
{
    clock_t start_time = clock();
    FFsamples samples = new_samples(nips);
    for (int i = 0; i < epochs; i++)
    {
        clock_t epoch_start_time = clock();
        printf("Epoch %d\n", i);
        log_info("Epoch %d", i);
        increase_indent();
        shuffle(data);
        double error = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            generate_samples(data, j, samples);
            error = train_ff_net(ffnet, samples.pos, samples.neg, rate);
        }
         printf("\tLoss %.12f :: learning rate %f\n",
             (double)error, // / i,
             (double)rate);
         double epoch_time = (double)(clock() - epoch_start_time) / CLOCKS_PER_SEC;
         printf("\tEpoch time: %.2f seconds\n", epoch_time);
        decrease_indent();
    }
    double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("\nTotal training time: %.2f seconds\n\n", total_time);
    free_samples(samples);
}

void evaluate(void)
{
    log_info("Testing FFNet...");
    initConfusionMatrix();
    for (int i = 0; i < 100; i++)
    {
        double *const in = data.in[i];
        double *const tg = data.tg[i];
        int gt = -1;
        for (int j = 0; j < data.num_class; j++)
        {
            if (tg[j] == 1.0f)
            {
                gt = j;
                break;
            }
        }
        const int pd = predict_ff_net(ffnet, in, nops, nips);
        addPrediction(gt, pd);
    }
    printf("\n");
    printNormalizedConfusionMatrix();
}

int main(void)
{
    setup();

    train_loop();
    log_info("Training done");

    printf("Testing...\n");
    evaluate();

    dfree(data);
    close_log_file();
    free_ff_net(ffnet);

    return 0;
}
