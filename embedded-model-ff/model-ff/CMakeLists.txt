idf_component_register(SRCS "ff-net/ff-net.c"
                        "ff-cell/ff-cell.c"
                        "adam/adam.c"
                        "utils/utils.c"
                        "data/data.c"
                        "ff-utils/ff-utils.c"
                        "losses/losses.c"
                       INCLUDE_DIRS "."
                       REQUIRES metrics)
