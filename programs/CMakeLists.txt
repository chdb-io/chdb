add_compile_options("$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${COVERAGE_FLAGS}>")

include(${ClickHouse_SOURCE_DIR}/cmake/split_debug_symbols.cmake)

configure_file (config_tools.h.in ${CONFIG_INCLUDE_PATH}/config_tools.h)

macro(clickhouse_target_link_split_lib target name)
    target_link_libraries(${target} PRIVATE clickhouse-${name}-lib)
endmacro()

macro(clickhouse_program_add_library name)
    string(TOUPPER ${name} name_uc)
    string(REPLACE "-" "_" name_uc ${name_uc})

    # Some dark magic
    set(CLICKHOUSE_${name_uc}_SOURCES ${CLICKHOUSE_${name_uc}_SOURCES} PARENT_SCOPE)
    set(CLICKHOUSE_${name_uc}_LINK ${CLICKHOUSE_${name_uc}_LINK} PARENT_SCOPE)
    set(CLICKHOUSE_${name_uc}_INCLUDE ${CLICKHOUSE_${name_uc}_INCLUDE} PARENT_SCOPE)

    add_library(clickhouse-${name}-lib ${CLICKHOUSE_${name_uc}_SOURCES})

    set(_link ${CLICKHOUSE_${name_uc}_LINK}) # can't use ${} in if()
    if(_link)
        target_link_libraries(clickhouse-${name}-lib ${CLICKHOUSE_${name_uc}_LINK})
    endif()

    target_include_directories (clickhouse-${name}-lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set(_include ${CLICKHOUSE_${name_uc}_INCLUDE}) # can't use ${} in if()
    if (_include)
        target_include_directories(clickhouse-${name}-lib ${CLICKHOUSE_${name_uc}_INCLUDE})
    endif()
endmacro()

macro(clickhouse_program_add name)
    clickhouse_program_add_library(${name})
endmacro()

add_subdirectory (bash-completion)
add_subdirectory (benchmark)
add_subdirectory (check-marks)
add_subdirectory (checksum-for-compressed-block)
add_subdirectory (client)
add_subdirectory (compressor)
add_subdirectory (disks)
add_subdirectory (extract-from-config)
add_subdirectory (format)
add_subdirectory (fst-dump-tree)
add_subdirectory (git-import)
add_subdirectory (install)
add_subdirectory (keeper-bench)
add_subdirectory (keeper-data-dumper)
add_subdirectory (keeper-utils)
add_subdirectory (local)
add_subdirectory (obfuscator)
add_subdirectory (server)
add_subdirectory (static-files-disk-uploader)
add_subdirectory (su)
add_subdirectory (zookeeper-dump-tree)
add_subdirectory (zookeeper-remove-by-list)

clickhouse_add_executable (clickhouse main.cpp)

# A library that prevent usage of several functions from libc.
if (ARCH_AMD64 AND OS_LINUX AND NOT OS_ANDROID)
    set (HARMFUL_LIB harmful)
endif ()

target_link_libraries (clickhouse PRIVATE clickhouse_common_io ${HARMFUL_LIB})
target_include_directories (clickhouse PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Link system iconv library on macOS for libxml2 compatibility
if (APPLE)
    target_link_libraries (clickhouse PRIVATE iconv)
endif()

macro(clickhouse_program_alias alias)
    message(STATUS "Adding alias ${alias}")
    add_custom_command (TARGET clickhouse POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink clickhouse ${alias})
    install (FILES "${CMAKE_CURRENT_BINARY_DIR}/${alias}" DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT clickhouse)
endmacro()

function(clickhouse_program_install name lib_name)
    clickhouse_target_link_split_lib(clickhouse ${lib_name})
    add_custom_command (TARGET clickhouse POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink clickhouse ${name})
    install (FILES "${CMAKE_CURRENT_BINARY_DIR}/${name}" DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT clickhouse)

    foreach(alias ${ARGN})
        clickhouse_program_alias(${alias})
    endforeach()
endfunction()
