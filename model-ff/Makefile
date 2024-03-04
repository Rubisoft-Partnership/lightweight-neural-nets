BIN = target/main.out
PROJECT_BASEPATH = $(realpath .)

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native -Ilib -DPROJECT_BASEPATH=\"$(PROJECT_BASEPATH)\"

ifeq ($(DATASET),)
	DATASET_DEF =
else ifeq ($(DATASET),mnist)
	DATASET_DEF = -DATA_MNIST
else ifeq ($(DATASET),digits)
	DATASET_DEF = -DATA_DIGITS
endif

LDFLAGS = -lm

CC = gcc

SRC = src/main.c lib/ff-net/ff-net.c lib/ff-cell/ff-cell.c lib/logging/logging.c lib/data/data.c lib/utils/utils.c

all:
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(DATASET_DEF) $(LDFLAGS)

run:
	./$(BIN)

clean:
	rm -f $(BIN)