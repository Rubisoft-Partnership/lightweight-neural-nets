# Basic Configuration
BIN_PATH = target
BIN_FILE = main.out
BUILD_PATH = build
LIB_FOLDER = lib
PROJECT_BASEPATH = $(realpath .)

# Compilers
GCC = gcc
GPP = g++

# Compiler Flags
CFLAGS = -Wall -Wextra -pedantic -std=c99 -Ilib
CPPFLAGS = -Wall -Wextra -pedantic -std=c++17 -Ilib -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
CPPFLAGS_NO_WARNINGS = -std=c++17 -Ilib -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE

METRICS_BASEPATH = $(PROJECT_BASEPATH)/../metrics
MODELFF_BASEPATH = $(PROJECT_BASEPATH)/../model-ff
MODELBP_BASEPATH = $(PROJECT_BASEPATH)/../model-bp

# Metrics Source Files
METRICS_SRC = $(wildcard $(METRICS_BASEPATH)/$(LIB_FOLDER)/*/*.c) $(METRICS_BASEPATH)/$(LIB_FOLDER)/metrics.c $(METRICS_BASEPATH)/$(LIB_FOLDER)/metrics.cpp
METRICS_OBJ = $(addprefix $(BUILD_PATH)/, $(notdir $(patsubst %.c,%.o,$(filter %.c,$(METRICS_SRC))))) $(BUILD_PATH)/cmetrics.o

# ModelFF Source Files
MODELFF_SRC = $(wildcard $(MODELFF_BASEPATH)/$(LIB_FOLDER)/*/*.c) $(MODELFF_BASEPATH)/model-ff.cpp
MODELFF_OBJ = $(addprefix $(BUILD_PATH)/, $(notdir $(patsubst %.c,%.o,$(filter %.c,$(MODELFF_SRC))))) $(BUILD_PATH)/model-ff.o

# ModelBP Source Files
MODELBP_SRC = $(MODELBP_BASEPATH)/model-bp.cpp
MODELBP_OBJ = $(addprefix $(BUILD_PATH)/, $(notdir $(patsubst %.cpp,%.o,$(MODELBP_SRC))))

#Include Paths
METRICS_INCLUDE = -I $(METRICS_BASEPATH)/$(LIB_FOLDER)
FRAMEWORK_INCLUDE = -I $(PROJECT_BASEPATH)/$(LIB_FOLDER)
SPDLOG_INCLUDE = -I $(PROJECT_BASEPATH)/$(LIB_FOLDER)/logging/include
MODELFF_INCLUDE = -I $(MODELFF_BASEPATH)/$(LIB_FOLDER)
BASE_INCLUDE = -I $(PROJECT_BASEPATH)/..
TDNN_INCLUDE = -I $(PROJECT_BASEPATH)/../tiny-dnn

# Dependencies
DEPS = src/main.cpp
DEPS_OBJ = $(patsubst src/%.cpp,$(BUILD_PATH)/%.o,$(DEPS))
FRAMEWORK_SRC = $(wildcard $(PROJECT_BASEPATH)/$(LIB_FOLDER)/*/*.cpp)
FRAMEWORK_OBJ = $(addprefix $(BUILD_PATH)/, $(notdir $(patsubst %.cpp,%.o,$(FRAMEWORK_SRC))))

# Object Files
OBJ_FILES = $(FRAMEWORK_OBJ) $(METRICS_OBJ) $(DEPS_OBJ) $(MODELFF_OBJ) $(MODELBP_OBJ)

# Default target
all: $(BIN_PATH)/$(BIN_FILE)

# Metrics C files Compilation
$(METRICS_OBJ): $(METRICS_SRC)
	@echo "Compiling Metrics C files..."
	@for file in $(filter %.c,$(METRICS_SRC)); do \
    	obj_file=$$(basename $$file .c).o; \
    	$(GCC) -c $$file $(CFLAGS) $(METRICS_INCLUDE) -o $(BUILD_PATH)/$$obj_file; \
		echo "Compiled $$file"; \
	done
	$(GPP) -c $(METRICS_BASEPATH)/$(LIB_FOLDER)/metrics.cpp $(CPPFLAGS) $(METRICS_INCLUDE) -o $(BUILD_PATH)/cmetrics.o

# ModelFF C files Compilation
$(MODELFF_OBJ): $(MODELFF_SRC)
	@echo "Compiling Model C/C++ files..."
	@for file in $(filter %.c,$(MODELFF_SRC)); do \
    	obj_file=$$(basename $$file .c).o; \
    	$(GCC) -c $$file $(CFLAGS) $(MODELFF_INCLUDE) $(METRICS_INCLUDE) -o $(BUILD_PATH)/$$obj_file; \
		echo "Compiled $$file"; \
	done
	$(GPP) -c $(PROJECT_BASEPATH)/../model-ff/model-ff.cpp $(CPPFLAGS) $(MODELFF_INCLUDE) $(METRICS_INCLUDE) -o $(BUILD_PATH)/model-ff.o;

# ModelBP C++ files Compilation
$(MODELBP_OBJ): $(MODELBP_SRC)
	@echo $(TDNN_INCLUDE)
	@echo "Compiling Model C++ files..."
	@for file in $(MODELBP_SRC); do \
		obj_file=$$(basename $$file .cpp).o; \
		$(GPP) -c $$file $(CPPFLAGS_NO_WARNINGS) $(METRICS_INCLUDE) $(TDNN_INCLUDE) $(SPDLOG_INCLUDE) -o $(BUILD_PATH)/$$obj_file; \
		echo "Compiled $$file"; \
	done

# Framework C++ files Compilation
$(FRAMEWORK_OBJ): $(FRAMEWORK_SRC)
	@echo "Compiling Framework C++ files..."
	@for file in $(FRAMEWORK_SRC); do \
    	obj_file=$$(basename $$file .cpp).o; \
    	$(GPP) -c $$file $(CPPFLAGS) $(FRAMEWORK_INCLUDE) $(BASE_INCLUDE) $(MODELFF_INCLUDE) $(SPDLOG_INCLUDE) $(METRICS_INCLUDE) $(TDNN_INCLUDE) -o $(BUILD_PATH)/$$obj_file; \
		echo "Compiled $$file"; \
	done

# Main script Compilation
$(DEPS_OBJ): $(DEPS)
	@mkdir -p $(BUILD_PATH) # Ensure the build directory exists
	$(GPP) -c $< -o $@ $(CPPFLAGS) $(FRAMEWORK_INCLUDE) $(BASE_INCLUDE) $(MODELFF_INCLUDE) $(TDNN_INCLUDE) $(METRICS_INCLUDE) $(SPDLOG_INCLUDE)
	echo "Compiled $<"

# Linking all object files to create the executable
$(BIN_PATH)/$(BIN_FILE): $(DEPS_OBJ) $(FRAMEWORK_OBJ) $(METRICS_OBJ) $(MODELFF_OBJ) $(MODELBP_OBJ)
	@echo "Linking all object files..."
	@mkdir -p $(BIN_PATH) # Ensure the bin directory exists
	$(GPP) -o $@ $^

# Clean target and object files
clean:
	rm -f $(BIN_PATH)/$(BIN_FILE)
	rm -f $(OBJ_FILES)
