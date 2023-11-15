CC = g++
CFLAGS = -I ../tiny-dnn
DEPS = src/run.cpp
EXECNAME = target/main.out

all: main

main: $(DEPS)
	$(CC) -o $(EXECNAME)  $^ $(CFLAGS)

clean:
	rm -f *.out