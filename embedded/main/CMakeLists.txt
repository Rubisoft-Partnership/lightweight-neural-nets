idf_component_register(SRCS "main.cpp"
                    PRIV_REQUIRES spi_flash
                    REQUIRES driver
                    PRIV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../tiny-dnn"
                    REQUIRES model-bp
                    INCLUDE_DIRS ".")

target_compile_options(${COMPONENT_LIB} PRIVATE -frtti -fexceptions)
