# Build the latency benchmark for one generated network on the STM32H7B3I-DK.
#   make GEN=../build/gen/<tag>        -> ../build/h7b3/<tag>/fw.elf
# Startup, system file, HAL configuration and linker script come from ST's STM32H7B3I-DK
# template in STM32CubeH7 1.13.0 (installed by STM32CubeMX); nothing is copied into the repo.

TOOLS ?= C:/ST/STM32CubeIDE_2.1.1/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/tools/bin
STAI ?= C:/ST/STEdgeAI/4.0/Middlewares/ST/AI
CUBE ?= C:/Users/<user>/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0
TPL := $(CUBE)/Projects/STM32H7B3I-DK/Templates
HAL := $(CUBE)/Drivers/STM32H7xx_HAL_Driver
GEN ?= ../build/gen/highd_cls_model_1790282740507531593_621856_int8_io
BUILD ?= ../build/h7b3/$(notdir $(GEN))
N_RUNS ?= 200

CC := $(TOOLS)/arm-none-eabi-gcc
SIZE := $(TOOLS)/arm-none-eabi-size

ARCH := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard
CFLAGS := $(ARCH) -O2 -g -Wall -ffunction-sections -fdata-sections -DSTM32H7B3xxQ -DUSE_HAL_DRIVER \
          -DN_RUNS=$(N_RUNS) -I$(CUBE)/Drivers/CMSIS/Include -I$(CUBE)/Drivers/CMSIS/Device/ST/STM32H7xx/Include \
          -I$(HAL)/Inc -I$(TPL)/Inc -I$(STAI)/Inc -I$(GEN)
LDFLAGS := $(ARCH) -T $(TPL)/STM32CubeIDE/STM32H7B3LIHXQ_FLASH.ld -Wl,--gc-sections -Wl,-Map=$(BUILD)/fw.map \
           --specs=nano.specs --specs=nosys.specs \
           -Wl,--start-group $(STAI)/Lib/GCC/ARMCortexM7/NetworkRuntime1200_CM7_GCC.a -lc -lm -Wl,--end-group

SRCS := main.c $(TPL)/Src/system_stm32h7xx.c \
        $(HAL)/Src/stm32h7xx_hal.c $(HAL)/Src/stm32h7xx_hal_rcc.c $(HAL)/Src/stm32h7xx_hal_rcc_ex.c \
        $(HAL)/Src/stm32h7xx_hal_pwr.c $(HAL)/Src/stm32h7xx_hal_pwr_ex.c $(HAL)/Src/stm32h7xx_hal_cortex.c \
        $(wildcard $(GEN)/*.c)
ASRC := $(CUBE)/Drivers/CMSIS/Device/ST/STM32H7xx/Source/Templates/gcc/startup_stm32h7b3xxq.s

all: $(BUILD)/fw.elf

$(BUILD)/fw.elf: $(SRCS) $(ASRC) Makefile
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SRCS) $(ASRC) $(LDFLAGS) -o $@
	$(SIZE) $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
