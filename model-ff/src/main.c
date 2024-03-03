#include <time.h>
#include <string.h>
#include <stdlib.h>

#include <ff-lib/ff-lib.h>
#include <data/data.h>
#include <logging/logging.h>


int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
    int nips = DATA_FEATURES;
    const int nops = DATA_CLASSES;
    int layers_sizes[] = {DATA_FEATURES, 100, 100};

    const int layers_number = sizeof(layers_sizes) / sizeof(layers_sizes[0]);
    // Hyper Parameters.
    double rate = 0.5f;
    const double anneal = 0.99f;
    const int iterations = 60;
    const double threshold = 4.0f;

    Data data;

    open_log_file_with_timestamp();
    data = build();

    set_log_level(LOG_DEBUG);

    const FFNet ffnet = ffnetbuild(layers_sizes, layers_number, relu, pdrelu, threshold);
    FFsamples samples = new_samples(nips);
    for (int i = 0; i < iterations; i++)
    {
        log_info("Iteration %d", i);
        shuffle(data);
        double error = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            log_debug("Sample %d", j);
            generate_samples(data, j, samples);

            error += fftrainnet(ffnet, samples.pos, samples.neg, rate);
            log_debug("Error %f", error);
        }
        printf("error %.12f :: learning rate %f\n",
               (double)error / data.rows,
               (double)rate);
        rate *= anneal;
    }

    log_info("Training done");

    printf("Predictions:\n");

    for (int i = 0; i < 10; i++)
    {
        double *const in = data.in[i];
        double *const tg = data.tg[i];
        const int pd = ffpredictnet(ffnet, in, nops, nips);
        // Prints target.
        for (int i = 0; i < data.num_class; i++)
        {
            if (tg[i] == 1.0f){
                printf("GT: %d, prediction:%d\n", i, pd);
                break;
            }
        }
    }

    dfree(data);

    close_log_file();
    return 0;
}
