# Build the latency benchmark for one generated network on the NUCLEO-F401RE.
#   make GEN=../build/gen/<tag>        -> ../build/f401/<tag>/fw.elf
#   make IDLE=1 (core asleep) or IDLE=2 (core spinning) -> ../build/f401/idle<n>/fw.elf, power baselines
# Tool locations default to this machine's STM32CubeIDE 2.1.1 and ST Edge AI Core 4.0 installs.

TOOLS ?= C:/ST/STM32CubeIDE_2.1.1/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/tools/bin
STAI ?= C:/ST/STEdgeAI/4.0/Middlewares/ST/AI
CMSIS_CORE ?= C:/Users/<user>/STM32Cube/Repository/STM32Cube_FW_H7_V1.13.0/Drivers/CMSIS/Include
DEV ?= ../third_party/cmsis-device-f4
GEN ?= ../build/gen/highd_cls4k_int8_io
N_RUNS ?= 200
IDLE ?=
ifeq ($(IDLE),)
BUILD ?= ../build/f401/$(notdir $(GEN))
IDLEDEF :=
else
BUILD ?= ../build/f401/idle$(IDLE)
IDLEDEF := -DIDLE_MODE=$(IDLE)
endif

CC := $(TOOLS)/arm-none-eabi-gcc
SIZE := $(TOOLS)/arm-none-eabi-size

ARCH := -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS := $(ARCH) -O2 -g -Wall -ffunction-sections -fdata-sections -DSTM32F401xE -DN_RUNS=$(N_RUNS) $(IDLEDEF) \
          -I$(CMSIS_CORE) -I$(DEV)/Include -I$(STAI)/Inc -I$(GEN)
LDFLAGS := $(ARCH) -T STM32F401RETx_FLASH.ld -Wl,--gc-sections -Wl,-Map=$(BUILD)/fw.map \
           --specs=nano.specs --specs=nosys.specs \
           -Wl,--start-group $(STAI)/Lib/GCC/ARMCortexM4/NetworkRuntime1200_CM4_GCC.a -lc -lm -Wl,--end-group

SRCS := main.c $(DEV)/Source/Templates/system_stm32f4xx.c $(wildcard $(GEN)/*.c)
ASRC := $(DEV)/Source/Templates/gcc/startup_stm32f401xe.s

all: $(BUILD)/fw.elf

$(BUILD)/fw.elf: $(SRCS) $(ASRC) STM32F401RETx_FLASH.ld Makefile
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(SRCS) $(ASRC) $(LDFLAGS) -o $@
	$(SIZE) $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
