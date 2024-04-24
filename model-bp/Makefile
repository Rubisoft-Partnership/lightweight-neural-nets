# Basic Configuration
BIN_PATH = target
BIN_FILE = main.out
BUILD_PATH = build
PROJECT_BASEPATH = $(realpath .)

# Compilers
GCC = gcc
GPP = g++

# Compiler Flags
CFLAGS = -Wall -Wextra -pedantic -std=c99 -Ilib
CPPFLAGS = -Wall -Wextra -pedantic -std=c++14 -Ilib

METRICS_BASEPATH = $(PROJECT_BASEPATH)/../metrics

# Metrics Source Files
METRICS_SRC = $(METRICS_BASEPATH)/lib/confusion-matrix/confusion-matrix.c \
              $(METRICS_BASEPATH)/lib/accuracy/accuracy.c \
              $(METRICS_BASEPATH)/lib/predictions/predictions.c
METRICS_OBJ = $(addprefix $(BUILD_PATH)/, $(notdir $(patsubst %.c,%.o,$(METRICS_SRC))))

#Include Paths
METRICS_INCLUDE = -I $(METRICS_BASEPATH)/lib
TDNN_INCLUDE = -I ../tiny-dnn

# Dependencies
DEPS = src/run.cpp
TDNN_OBJ = $(patsubst src/%.cpp,$(BUILD_PATH)/%.o,$(DEPS))

# Object Files
OBJ_FILES = $(TDNN_OBJ) $(METRICS_OBJ)

# Default target
all: $(BIN_PATH)/$(BIN_FILE)

# Metrics C files Compilation
$(METRICS_OBJ): $(METRICS_SRC)
	@echo "Compiling Metrics C files..."
	@for file in $(METRICS_SRC); do \
    	obj_file=$$(basename $$file .c).o; \
    	$(GCC) -c $$file $(CFLAGS) $(METRICS_INCLUDE) -o $(BUILD_PATH)/$$obj_file; \
	done

# Tiny-DNN C++ files Compilation
$(TDNN_OBJ): $(DEPS)
	@mkdir -p $(BUILD_PATH) # Ensure the build directory exists
	$(GPP) -c $< -o $@ $(CPPFLAGS) $(TDNN_INCLUDE) $(METRICS_INCLUDE)

# Linking all object files to create the executable
$(BIN_PATH)/$(BIN_FILE): $(TDNN_OBJ) $(METRICS_OBJ)
	@echo "Linking all object files..."
	@mkdir -p $(BIN_PATH) # Ensure the bin directory exists
	$(GPP) -o $@ $^

# Clean target and object files
clean:
	rm -f $(BIN_PATH)/$(BIN_FILE)
	rm -f $(OBJ_FILES)

