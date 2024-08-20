idf_component_register(SRCS "model.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES metrics model-ff driver esp_timer)
