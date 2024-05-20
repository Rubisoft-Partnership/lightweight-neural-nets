BIN_PATH = target
BIN_FILE = main.out
PROJECT_BASEPATH = $(realpath .)

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native -Ilib -DPROJECT_BASEPATH=\"$(PROJECT_BASEPATH)\" -g

ifeq ($(DATASET),)
	DATASET_DEF =
else ifeq ($(DATASET),mnist)
	DATASET_DEF = -DDATA_MNIST
else ifeq ($(DATASET),digits)
	DATASET_DEF = -DDATA_DIGITS
endif

LDFLAGS = -lm

CC = gcc

METRICS_BASEPATH = $(PROJECT_BASEPATH)/../metrics

# FF library
SRC = src/main.c lib/ff-net/ff-net.c lib/ff-cell/ff-cell.c lib/logging/logging.c lib/data/data.c lib/utils/utils.c lib/adam/adam.c lib/losses/losses.c lib/ff-utils/ff-utils.c

# Metrics library
METRICS_SRC = $(wildcard $(METRICS_BASEPATH)/lib/*/*.c) $(METRICS_BASEPATH)/lib/metrics.c
SRC += $(METRICS_SRC)

INCLUDE = -I$(METRICS_BASEPATH)/lib

all:
	@mkdir -p $(BIN_PATH)
	$(CC) -o $(BIN_PATH)/$(BIN_FILE) $(SRC) $(CFLAGS) $(DATASET_DEF) $(LDFLAGS) $(INCLUDE)

run:
	./$(BIN_PATH)/$(BIN_FILE)

clean:
	rm -f $(BIN_PATH)/$(BIN_FILE)