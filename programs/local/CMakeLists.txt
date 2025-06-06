set (CLICKHOUSE_LOCAL_SOURCES
    chdb.cpp
    LocalServer.cpp
)

if (USE_PYTHON)
    set (CHDB_SOURCES
        chdb.cpp
        FormatHelper.cpp
        ListScan.cpp
        LocalChdb.cpp
        LocalServer.cpp
        NumpyType.cpp
        PandasAnalyzer.cpp
        PandasDataFrame.cpp
        PandasScan.cpp
        PybindWrapper.cpp
        PythonConversion.cpp
        PythonDict.cpp
        PythonReader.cpp
        PythonTableCache.cpp
        PythonImportCache.cpp
        PythonImporter.cpp
        PythonSource.cpp
        PythonUtils.cpp
        StoragePython.cpp
        TableFunctionPython.cpp
    )
    set (CLICKHOUSE_LOCAL_SOURCES ${CLICKHOUSE_LOCAL_SOURCES} ${CHDB_SOURCES})

    # include path from shell cmd "python3 -m pybind11 --includes"
    execute_process(COMMAND python3 -m pybind11 --includes
        OUTPUT_VARIABLE PYBIND11_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE ".*-I([^ ]+).*" "\\1" PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDES})
    include_directories(${PYBIND11_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

    # include Python.h
    execute_process(COMMAND python3-config --includes
        OUTPUT_VARIABLE PYTHON_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE ".*-I([^ ]+).*" "\\1" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDES})

    foreach(_file ${CHDB_SOURCES})
        set_source_files_properties(${_file}
            PROPERTIES INCLUDE_DIRECTORIES
            ${PYTHON_INCLUDE_DIR}
        )
    endforeach(_file)

    # get python version, something like python3.x
    execute_process(COMMAND python3 -c "import sys; print('python3.'+str(sys.version_info[1]))"
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # remove all warning, because pybind11 will generate a lot of warning
    if (OS_LINUX)
        # pybind11 will try to find x86_64-linux-gnu/${PYTHON_VERSION}/pyconfig.h
        # use -idirafter to make it find the right one and not polute the include path
        # set_source_files_properties(LocalChdb.cpp PROPERTIES COMPILE_FLAGS
        #     "-w -idirafter /usr/include -include x86_64-linux-gnu/${PYTHON_VERSION}/pyconfig.h"
        # )
        if (PYTHON_VERSION STREQUAL "python3.6" OR PYTHON_VERSION STREQUAL "python3.7" OR PYTHON_VERSION STREQUAL "python3.8")
            foreach(_file ${CHDB_SOURCES})
                set_source_files_properties(${_file}
                    PROPERTIES COMPILE_FLAGS
                    "-w -idirafter /usr/include -include crypt.h"
                )
            endforeach(_file)
        else()
            foreach(_file ${CHDB_SOURCES})
                set_source_files_properties(${_file}
                    PROPERTIES COMPILE_FLAGS
                    "-w"
                )
            endforeach(_file)
        endif()
    elseif (OS_DARWIN)
        foreach(_file ${CHDB_SOURCES})
            set_source_files_properties(${_file}
                PROPERTIES COMPILE_FLAGS
                "-w"
            )
        endforeach(_file)
    endif()
endif()

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
        ch_contrib::parquet
)

clickhouse_program_add(local)

target_link_libraries(clickhouse-local-lib PRIVATE clickhouse-server-lib)

if (TARGET ch_rust::skim)
    target_link_libraries(clickhouse-local-lib PRIVATE ch_rust::skim)
endif()
if (TARGET ch_contrib::azure_sdk)
    target_link_libraries(clickhouse-local-lib PRIVATE ch_contrib::azure_sdk)
endif()
if (TARGET ch_contrib::utf8proc)
    target_link_libraries(clickhouse-local-lib PRIVATE ch_contrib::utf8proc)
endif()

# Always use internal readpassphrase
target_link_libraries(clickhouse-local-lib PRIVATE readpassphrase)
