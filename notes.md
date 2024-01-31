# Developing notes

## Advancements

- Changed `Tinn` structure:
  - Added `act` and `pdact` fields
  - Added `Treshold` parameter (theta)
- Added ReLU activation function and its derivative
- Added Sigmoid activation function and its derivative
- File structure:
  - `ff-lib.c` constains ff-functions and Tinn functions
  - `ff-lib.h` contains Tinn struct, externalized functions (both Tinn ones and ff ones)
- Makefile

## TODO

- [ ] Fix how the files are organized (library and decide how to use Tinn functions)
- [x] Enable multiple layers
- [x] ReLU doesn't work becuase the pd can be 0
- [ ] Optimizer
- [ ] Find decent hyperparameters
- [ ] Add SymBa Loss
- [ ] Add loss parametrization
- [ ] Return list of errors for each layer
- [ ] Free function Tinn array

