#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"

#include "model.hpp"

extern "C"
{
    void app_main(void);
}

void app_main(void)
{
    ModelFF model;
    printf("Heap size: %d\n", xPortGetFreeHeapSize());

    model.build();

    model.train(1, 2, 0.005);
}
