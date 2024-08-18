# Register the main source files of the metrics component
idf_component_register(SRCS "lib/metrics.cpp"
                       INCLUDE_DIRS "lib/."
                       PRIV_INCLUDE_DIRS "lib/accuracy" "lib/f1-score" "lib/precision" "lib/recall" "lib/predictions" "lib/confusion-matrix")

