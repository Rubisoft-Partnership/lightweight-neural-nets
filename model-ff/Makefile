BIN = target/main.out

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native -Ilib

ifeq ($(dataset),)
	DATASET_DEF =
else ifeq ($(dataset),mnist)
	DATASET_DEF = -DATA_MNIST
else ifeq ($(dataset),digits)
	DATASET_DEF = -DATA_DIGITS
endif

LDFLAGS = -lm

CC = gcc

SRC = src/main.c lib/ff-lib/ff-lib.c lib/logging/logging.c lib/data/data.c lib/utils/utils.c

all:
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(DATASET_DEF) $(LDFLAGS)

run:
	./$(BIN)

clean:
	rm -f $(BIN)