BIN = target/main.out

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

LDFLAGS = -lm

CC = gcc

IPATH = -I../tinn

SRC = src/main.c ../tinn/Tinn.c

all:
	if [ ! -d "target" ]; then mkdir target; fi
	$(CC) -o $(BIN) $(SRC) $(CFLAGS) $(LDFLAGS) $(IPATH)

run:
	./$(BIN)

clean:
	rm -f $(BIN)
