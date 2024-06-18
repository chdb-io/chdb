set(LIBRARY_DIR "${ClickHouse_SOURCE_DIR}/contrib/utf8proc")

set(SRCS
    "${LIBRARY_DIR}/utf8proc.c"
)

add_library(utf8proc ${SRCS})
target_include_directories(utf8proc SYSTEM PUBLIC "${LIBRARY_DIR}")
add_library(ch_contrib::utf8proc ALIAS utf8proc)
