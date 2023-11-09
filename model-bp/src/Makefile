CC = g++
CFLAGS = -I ../../tiny-dnn
DEPS = run.cpp
EXECNAME = main.out

all: main

main: $(DEPS)
	$(CC) -o $(EXECNAME)  $^ $(CFLAGS)

clean:
	rm -f *.out