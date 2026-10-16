/*
 * Latency benchmark for one ST Edge AI network (legacy C API) on the STM32H7B3I-DK.
 *
 * Same measurement as firmware/f401/main.c: DWT cycle counter around ai_network_run,
 * fresh pseudo-random inputs, 5 warm-up and N_RUNS timed runs, no interrupts while timing.
 * Clock, power and caches follow ST's STM32H7B3I-DK template (STM32CubeH7 1.13.0):
 * direct SMPS supply (set in the template's system_stm32h7xx.c), VOS0, HSE 24 MHz,
 * PLL1 280 MHz, flash latency 6, I- and D-cache on. Data and the activation buffer sit in
 * DTCM (template linker script).
 *
 * The result block g_result (magic, core_hz, runs, cyc_min, cyc_mean, cyc_max, status,
 * loop counter) is read over SWD by unas/local_bench.py. After the timed runs the board
 * keeps running inferences (LED toggles every 1000 runs) for current measurements.
 */
#include <stdint.h>
#include "stm32h7xx_hal.h"
#include "network.h"
#include "network_data.h"

#ifndef N_RUNS
#define N_RUNS 200
#endif
#ifndef N_WARMUP
#define N_WARMUP 5
#endif

#define RESULT_MAGIC 0x42454E43u /* "BENC" */
volatile uint32_t g_result[8] __attribute__((used));

#ifdef ACT_AXI
/* Activation buffer at the start of AXI SRAM (D-cached, 1 MiB, unused by the template linker
   script, which keeps all data in DTCM). Selected with make AXI=1. */
static ai_handle activations[] = {(ai_handle)0x24000000u};
#else
AI_ALIGNED(32) static ai_u8 pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];
static ai_handle activations[] = {pool0};
#endif
static ai_handle network = AI_HANDLE_NULL;

void SysTick_Handler(void) { HAL_IncTick(); }

static void fail(void)
{
    for (;;) {}
}

/* Verbatim from Projects/STM32H7B3I-DK/Templates/Src/main.c (280 MHz from HSE). */
static void SystemClock_Config(void)
{
    RCC_ClkInitTypeDef RCC_ClkInitStruct;
    RCC_OscInitTypeDef RCC_OscInitStruct;

    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.HSIState = RCC_HSI_OFF;
    RCC_OscInitStruct.CSIState = RCC_CSI_OFF;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 12;
    RCC_OscInitStruct.PLL.PLLN = 280;
    RCC_OscInitStruct.PLL.PLLFRACN = 0;
    RCC_OscInitStruct.PLL.PLLP = 2;
    RCC_OscInitStruct.PLL.PLLR = 2;
    RCC_OscInitStruct.PLL.PLLQ = 2;
    RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
    RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
        fail();

    RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_D1PCLK1 |
                                   RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1);
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
    RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_6) != HAL_OK)
        fail();
}

static uint32_t lcg = 12345u;
static uint32_t rnd(void)
{
    lcg = lcg * 1664525u + 1013904223u;
    return lcg;
}

static void fill_input(ai_buffer *in)
{
    if (AI_BUFFER_FMT_GET_TYPE(in[0].format) == AI_BUFFER_FMT_TYPE_FLOAT) {
        float *x = (float *)in[0].data;
        for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++)
            x[i] = (float)(rnd() >> 8) * (1.0f / 16777216.0f);
    } else {
        int8_t *x = (int8_t *)in[0].data;
        for (int i = 0; i < AI_NETWORK_IN_1_SIZE_BYTES; i++)
            x[i] = (int8_t)(rnd() >> 24);
    }
}

int main(void)
{
    SCB_EnableICache();
    SCB_EnableDCache();
    HAL_Init();
    SystemClock_Config();
    SystemCoreClockUpdate();
    HAL_SuspendTick(); /* no SysTick interrupts from here on */

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u; /* unlock DWT on the Cortex-M7 */
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    g_result[0] = 0;
    g_result[1] = SystemCoreClock;
#ifdef IDLE_MODE
    /* Baselines for power readings, same clock, caches and peripherals, no network:
       1 = core asleep (WFI, no interrupt enabled), 2 = core spinning on a RAM counter. */
    g_result[6] = 0x300u | IDLE_MODE;
    g_result[0] = RESULT_MAGIC;
    for (uint32_t k = 0;; k++) {
        if (IDLE_MODE == 1)
            __WFI();
        else
            g_result[7] = k;
    }
#endif
    ai_error err = ai_network_create_and_init(&network, activations, NULL);
    if (err.type != AI_ERROR_NONE) {
        g_result[6] = 0x100u | (uint32_t)err.type;
        g_result[0] = RESULT_MAGIC;
        fail();
    }
    ai_buffer *in = ai_network_inputs_get(network, NULL);
    ai_buffer *out = ai_network_outputs_get(network, NULL);

    uint32_t mn = 0xFFFFFFFFu, mx = 0;
    uint64_t sum = 0;
    for (int i = 0; i < N_WARMUP + N_RUNS; i++) {
        fill_input(in);
        uint32_t t0 = DWT->CYCCNT;
        ai_i32 nb = ai_network_run(network, in, out);
        uint32_t dt = DWT->CYCCNT - t0;
        if (nb != 1) {
            err = ai_network_get_error(network);
            g_result[6] = 0x200u | (uint32_t)err.type;
            g_result[0] = RESULT_MAGIC;
            fail();
        }
        if (i >= N_WARMUP) {
            if (dt < mn) mn = dt;
            if (dt > mx) mx = dt;
            sum += dt;
        }
    }
    g_result[2] = N_RUNS;
    g_result[3] = mn;
    g_result[4] = (uint32_t)(sum / N_RUNS);
    g_result[5] = mx;
    g_result[6] = 0;
    g_result[0] = RESULT_MAGIC;

    for (uint32_t k = 0;; k++) {
        fill_input(in);
        ai_network_run(network, in, out);
        g_result[7] = k;
    }
}
