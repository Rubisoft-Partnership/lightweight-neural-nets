#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"

extern "C"
{
    void app_main(void);
}

#define UART_NUM UART_NUM_0
#define BUF_SIZE (1024)

void app_main(void)
{
    // Configure UART parameters
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 122,  // Set a typical threshold
        .source_clk = UART_SCLK_APB, // Set the clock source, which is commonly used
        .flags = 0                   // Initialize any additional flags
    };
    uart_param_config(UART_NUM, &uart_config);

    // Install UART driver
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);

    uint8_t data[BUF_SIZE];

    printf("Waiting for data...\n");
    while (1)
    {
        // Read data from the UART
        int len = uart_read_bytes(UART_NUM, data, BUF_SIZE, 20 / portTICK_PERIOD_MS);
        if (len > 0)
        {
            // Write data back to the UART
            uart_write_bytes(UART_NUM, (const char *)data, len);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}
