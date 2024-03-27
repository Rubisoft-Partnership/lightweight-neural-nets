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

METRICS_BASEPATH = $(PROJECT_BASEPATH)/../metrics/lib

SRC = src/main.c lib/ff-net/ff-net.c lib/ff-cell/ff-cell.c lib/logging/logging.c lib/data/data.c lib/utils/utils.c $(METRICS_BASEPATH)/confusion-matrix/confusion-matrix.c

INCLUDE = -I$(METRICS_BASEPATH)

all:
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(DATASET_DEF) $(LDFLAGS) $(INCLUDE)

run:
	./$(BIN)

clean:
	rm -f $(BIN)