BIN_DIR = target
BUILD_DIR = build
LIB_OBJS = $(BUILD_DIR)/ff-lib.o
MNIST_BIN = $(BIN_DIR)/mnist.out
DIGITS_BIN = $(BIN_DIR)/digits.out

# Compiler and linker flags
CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native
LDFLAGS = -lm
CC = gcc
IPATH = -I ./src/

# Source files
MNIST_SRC = src/mnist.c
DIGITS_SRC = src/digits.c
LIB_SRC = src/ff-lib.c

# Object files, now placed in the build directory
MNIST_OBJ = $(MNIST_SRC:src/%.c=$(BUILD_DIR)/%.o)
DIGITS_OBJ = $(DIGITS_SRC:src/%.c=$(BUILD_DIR)/%.o)
LIB_OBJ = $(LIB_SRC:src/%.c=$(BUILD_DIR)/%.o)

# Default target
all: directories $(MNIST_BIN) $(DIGITS_BIN)

# Create the directories for binaries and build
directories:
	@if [ ! -d "$(BIN_DIR)" ]; then mkdir $(BIN_DIR); fi
	@if [ ! -d "$(BUILD_DIR)" ]; then mkdir $(BUILD_DIR); fi

# Compile the mnist binary
$(MNIST_BIN): $(MNIST_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(IPATH)

# Compile the digits binary
$(DIGITS_BIN): $(DIGITS_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(IPATH)

# Generic rule for compiling object files from source files
$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) $(IPATH) -c $< -o $@

# Clean up binaries and object files
clean:
	rm -f $(MNIST_BIN) $(DIGITS_BIN) $(MNIST_OBJ) $(DIGITS_OBJ) $(LIB_OBJ)
