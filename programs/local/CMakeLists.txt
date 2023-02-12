# set (CLICKHOUSE_LOCAL_SOURCES LocalServer.cpp)
set (CLICKHOUSE_LOCAL_SOURCES LocalServer.cpp LocalChdb.cpp)

# include path from shell cmd "python3 -m pybind11 --includes"
execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE ".*-I([^ ]+).*" "\\1" PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDES})
include_directories(${PYBIND11_INCLUDE_DIR})

# include Python.h
execute_process(COMMAND python3-config --includes
    OUTPUT_VARIABLE PYTHON_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE ".*-I([^ ]+).*" "\\1" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDES})
set_source_files_properties(LocalChdb.cpp PROPERTIES INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIR})

# remove all warning, because pybind11 will generate a lot of warning
set_source_files_properties(LocalChdb.cpp PROPERTIES COMPILE_FLAGS 
    "-w -idirafter /usr/include -include x86_64-linux-gnu/python3.9/pyconfig.h"
)

# add_library(clickhouse-local-lib SHARED ${CLICKHOUSE_LOCAL_SOURCES})

# target_link_libraries(clickhouse-local-lib
#     PRIVATE
#         boost::program_options
#         clickhouse_aggregate_functions
#         clickhouse_common_config
#         clickhouse_common_io
#         clickhouse_functions
#         clickhouse_parsers
#         clickhouse_storages_system
#         clickhouse_table_functions
# )

set (CLICKHOUSE_LOCAL_LINK
    PRIVATE
        boost::program_options
        clickhouse_aggregate_functions
        clickhouse_common_config
        clickhouse_common_io
        clickhouse_functions
        clickhouse_parsers
        clickhouse_storages_system
        clickhouse_table_functions
)

clickhouse_program_add(local)

if(NOT CLICKHOUSE_ONE_SHARED)
    target_link_libraries(clickhouse-local-lib PRIVATE clickhouse-server-lib)
endif()

# Always use internal readpassphrase
target_link_libraries(clickhouse-local-lib PRIVATE readpassphrase)

if (ENABLE_FUZZING)
    add_compile_definitions(FUZZING_MODE=1)
    set (WITH_COVERAGE ON)
    target_link_libraries(clickhouse-local-lib PRIVATE ${LIB_FUZZING_ENGINE})
endif ()
