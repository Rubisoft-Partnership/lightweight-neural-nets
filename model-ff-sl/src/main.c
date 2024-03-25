#include <time.h>
#include <stdlib.h>

#include <ff-net/ff-net.h>
#include <data/data.h>
#include <logging/logging.h>

// Input and output size is harded coded here as machine learning
// repositories usually don't include the input and output size in the data itself.
const int nips = DATA_FEATURES;
const int nops = DATA_CLASSES;
const int layers_sizes[] = {DATA_FEATURES, 1024, 1000, 500};

const int layers_number = sizeof(layers_sizes) / sizeof(layers_sizes[0]);
// Hyper Parameters.
double rate = 0.005;
const double anneal = 0.9999;
const int epochs = 5;
const double threshold = 4.0;

Data data;
FFNet ffnet;

static void setup(void)
{
    srand(0); // set random seed to 0
    set_log_level(LOG_DEBUG);
    open_log_file_with_timestamp();

    data = build();
    ffnet = ffnetbuild(layers_sizes, layers_number, relu, pdrelu, threshold);
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
        log_info("Epoch %d", i);
        increase_indent();
        shuffle(data);
        double error = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            generate_samples(data, j, samples);
            error = fftrainnet(ffnet, samples.pos, samples.neg, rate);
        }
         printf("Loss %.12f :: learning rate %f\n",
             (double)error, // / i,
             (double)rate);
         double epoch_time = (double)(clock() - epoch_start_time) / CLOCKS_PER_SEC;
         printf("Epoch time: %.2f seconds\n", epoch_time);
        rate *= anneal;
        decrease_indent();
    }
    double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Total training time: %.2f seconds\n", total_time);
    free_samples(samples);
}

void evaluate(void)
{
    log_info("Testing FFNet...");
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
        const int pd = ffpredictnet(ffnet, in, nops, nips);
        printf("Prediction: %d, ground truth: %d\n", pd, gt);
    }
}

int main(void)
{
    setup();

    train_loop();
    log_info("Training done");

    printf("Predictions:\n");
    evaluate();

    dfree(data);
    close_log_file();
    ffnetfree(ffnet);

    return 0;
}
