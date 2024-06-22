option(ENABLE_UTF8PROC "Enable UTF8PROC" 1)
if (NOT ENABLE_UTF8PROC)
    message(STATUS "Not using utf8proc")
    return()
endif()

set(LIBRARY_DIR "${ClickHouse_SOURCE_DIR}/contrib/utf8proc/")
set(UTF8PROC_INCLUDE_DIR "${LIBRARY_DIR}" CACHE STRING "Path to utf8proc")
message(STATUS "Using utf8proc from ${LIBRARY_DIR}")

set(SRCS
    "${LIBRARY_DIR}/utf8proc.c"
)

add_library(utf8proc ${SRCS})
add_library(ch_contrib::utf8proc ALIAS utf8proc)
target_include_directories(utf8proc PRIVATE "${LIBRARY_DIR}")
