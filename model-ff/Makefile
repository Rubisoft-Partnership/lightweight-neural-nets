BIN = target/main.out

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native -Ilib

# Check if the dataset variable is provided and append it to CFLAGS
ifeq ($(dataset),)
  DATASET_DEF =
else ifeq ($(dataset),MNIST)
	DATASET_DEF = -DMNIST=1
else ifeq ($(dataset),DIGITS)
	DATASET_DEF = -DDIGITS=1
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