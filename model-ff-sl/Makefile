BIN = target/main.out
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

SRC = src/main.c lib/ff-net/ff-net.c lib/ff-cell/ff-cell.c lib/logging/logging.c lib/data/data.c lib/utils/utils.c lib/adam/adam.c lib/losses/losses.c lib/ff-utils/ff-utils.c

# Metrics library
SRC += $(METRICS_BASEPATH)/lib/confusion-matrix/confusion-matrix.c $(METRICS_BASEPATH)/lib/accuracy/accuracy.c

INCLUDE = -I$(METRICS_BASEPATH)/lib

all:
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(DATASET_DEF) $(LDFLAGS) $(INCLUDE)

run:
	./$(BIN)

clean:
	rm -f $(BIN)