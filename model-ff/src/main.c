#include <time.h>
#include <string.h>
#include <stdlib.h>

#include <ff-lib/ff-lib.h>
#include <data/data.h>
#include <logging/logging.h>


// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{

    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
    int nips = 784;
    const int nops = 10;
    int layers_sizes[] = {784, 100, 100, 10};

    const int layers_number = sizeof(layers_sizes) / sizeof(layers_sizes[0]);
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate.
    // The number of iterations can be changed for stronger training.
    double rate = 0.5f;
    const double anneal = 0.99f;
    const int iterations = 60;
    const double threshold = 4.0f;

    Data data;
    if (DIGITS)
    {
        layers_sizes[0] = 64;
        nips = 64;
        data = build("../../dataset/digit_dataset/digits.txt", nips, nops);
        open_log_file_with_timestamp("../logs", "digits");
    }
    else if (MNIST)
    {
        open_log_file_with_timestamp("../logs", "mnist");
        data = build("../../dataset/mnist/mnist_train.txt", nips, nops);
    }

    set_log_level(LOG_DEBUG);

    // Train, baby, train.
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

    // This is how you save the neural network to disk.

    /// TODO: implement saving and freeing of the neural network
    /*
    xtsave(ffnet, "saved.tinn");
    xtfree(ffnet);
    */
    // This is how you load the neural network from disk.
    // const Tinn loaded = xtload("saved.tinn");
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).

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

    // xtprint(pd, data.num_class);
    // All done. Let's clean up.
    dfree(data);

    close_log_file();
    return 0;
}
