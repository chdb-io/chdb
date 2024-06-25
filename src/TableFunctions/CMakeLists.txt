include("${ClickHouse_SOURCE_DIR}/cmake/dbms_glob_sources.cmake")

add_headers_and_sources(clickhouse_table_functions .)

if (TARGET ch_contrib::hivemetastore)
    add_headers_and_sources(clickhouse_table_functions Hive)
endif ()

extract_into_parent_list(clickhouse_table_functions_sources dbms_sources
    ITableFunction.cpp
    TableFunctionView.cpp
    TableFunctionFactory.cpp
)
extract_into_parent_list(clickhouse_table_functions_headers dbms_headers
    ITableFunction.h
    TableFunctionView.h
    TableFunctionFactory.h
)

add_library(clickhouse_table_functions ${clickhouse_table_functions_headers} ${clickhouse_table_functions_sources})

target_link_libraries(clickhouse_table_functions PRIVATE clickhouse_parsers clickhouse_storages_system dbms)

if (USE_PYTHON)
    # Include path from shell cmd "python3 -m pybind11 --includes"
    execute_process(COMMAND python3 -m pybind11 --includes
        OUTPUT_VARIABLE PYBIND11_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Extract and set include directories specifically for source using pybind11
    string(REGEX MATCHALL "-I([^ ]+)" INCLUDE_DIRS_MATCHES ${PYBIND11_INCLUDES})
    set(PYTHON_INCLUDE_DIRS "")
    foreach(INCLUDE_DIR_MATCH ${INCLUDE_DIRS_MATCHES})
        string(REGEX REPLACE "-I" "" INCLUDE_DIR_MATCH ${INCLUDE_DIR_MATCH})
        # Accumulate all include directories
        set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS};${INCLUDE_DIR_MATCH}")
    endforeach()

    # Add include directories for pybind11
    set_source_files_properties(TableFunctionPython.cpp PROPERTIES INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")

    # get python version, something like python3.x
    execute_process(COMMAND python3 -c "import sys; print('python3.'+str(sys.version_info[1]))"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # remove all warning, because pybind11 will generate a lot of warning
    if (OS_LINUX)
        if (PYTHON_VERSION STREQUAL "python3.6" OR PYTHON_VERSION STREQUAL "python3.7" OR PYTHON_VERSION STREQUAL "python3.8")
            set_source_files_properties(TableFunctionPython.cpp PROPERTIES COMPILE_FLAGS
                "-w -idirafter /usr/include -include crypt.h"
            )
        else()
            set_source_files_properties(TableFunctionPython.cpp PROPERTIES COMPILE_FLAGS
                "-w"
            )
        endif()
    elseif (OS_DARWIN)
        set_source_files_properties(TableFunctionPython.cpp PROPERTIES COMPILE_FLAGS
            "-w"
        )
    endif()
endif()

if (TARGET ch_contrib::hivemetastore)
    target_link_libraries(clickhouse_table_functions PRIVATE ch_contrib::hivemetastore ch_contrib::hdfs ch_contrib::parquet)
endif ()

if (TARGET ch_contrib::azure_sdk)
    target_link_libraries(clickhouse_table_functions PRIVATE ch_contrib::azure_sdk)
endif ()

if (TARGET ch_contrib::simdjson)
    target_link_libraries(clickhouse_table_functions PRIVATE ch_contrib::simdjson)
endif ()

if (TARGET ch_contrib::utf8proc)
    target_link_libraries(clickhouse_table_functions PRIVATE ch_contrib::utf8proc)
    target_include_directories(clickhouse_table_functions PRIVATE ${UTF8PROC_INCLUDE_DIR})
endif ()

if (TARGET ch_contrib::rapidjson)
    target_link_libraries(clickhouse_table_functions PRIVATE ch_contrib::rapidjson)
endif ()
