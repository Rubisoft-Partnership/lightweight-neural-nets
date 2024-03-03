BIN = target/main.out

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

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

SRC = src/main.c src/ff-lib.c src/logging.c src/data.c src/utils.c

all:
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(DATASET_DEF) $(LDFLAGS)

run:
	./$(BIN)

clean:
	rm -f $(BIN)
