message("---------> Tiny-DNN Include Path: ${CMAKE_SOURCE_DIR}/../tiny-dnn")
idf_component_register(SRCS "model-bp.cpp"
                       INCLUDE_DIRS "."
                       PRIV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../tiny-dnn"
                       REQUIRES metrics)

target_compile_options(${COMPONENT_LIB} PRIVATE -frtti -fexceptions)
