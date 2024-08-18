idf_component_register(SRCS "main.cpp"
                    PRIV_REQUIRES spi_flash
                    PRIV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../model-ff/lib"
                    REQUIRES driver
                    REQUIRES model
                    INCLUDE_DIRS ".")
