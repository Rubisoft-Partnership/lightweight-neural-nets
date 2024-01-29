BIN = target/main.out

# CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

LDFLAGS = -lm

CC = gcc

IPATH = -I ./src/

SRC = src/main.c src/ff-lib.c

all:
	@if [ ! -d "target" ]; then mkdir target; fi
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(LDFLAGS) $(IPATH)

run:
	./$(BIN)

clean:
	rm -f $(BIN)
