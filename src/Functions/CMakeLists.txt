add_subdirectory(divide)

include("${ClickHouse_SOURCE_DIR}/cmake/dbms_glob_sources.cmake")
add_headers_and_sources(clickhouse_functions .)

add_library(clickhouse_functions_obj OBJECT ${clickhouse_functions_headers} ${clickhouse_functions_sources})
if (OMIT_HEAVY_DEBUG_SYMBOLS)
    target_compile_options(clickhouse_functions_obj PRIVATE "-g0")
endif()

list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_obj>)

list (APPEND PUBLIC_LIBS
        ch_contrib::wyhash
        ch_contrib::cityhash
        ch_contrib::farmhash
        clickhouse_dictionaries
        clickhouse_dictionaries_embedded
        clickhouse_parsers
        ch_contrib::consistent_hashing
        common
        dbms
        ch_contrib::metrohash
        ch_contrib::murmurhash
        ch_contrib::morton_nd
)

list (APPEND PRIVATE_LIBS
        ch_contrib::zlib
        ch_contrib::expected
        boost::filesystem
        divide_impl
        ch_contrib::xxHash
)

if (TARGET ch_rust::blake3)
    list (APPEND PUBLIC_LIBS
        ch_rust::blake3
    )
endif()

if (TARGET OpenSSL::Crypto)
    list (APPEND PUBLIC_LIBS OpenSSL::Crypto)
endif()

if (TARGET ch_contrib::icu)
    list (APPEND PRIVATE_LIBS ch_contrib::icu)
endif ()

if (TARGET ch_contrib::fastops)
    list (APPEND PRIVATE_LIBS ch_contrib::fastops)
endif ()

if (TARGET ch_contrib::llvm)
    list (APPEND PRIVATE_LIBS ch_contrib::llvm)
endif ()

if (TARGET ch_contrib::base64)
    list (APPEND PRIVATE_LIBS ch_contrib::base64)
endif()

if (ENABLE_NLP)
    list (APPEND PRIVATE_LIBS ch_contrib::cld2)
endif()

if (TARGET ch_contrib::sqids)
    list (APPEND PRIVATE_LIBS ch_contrib::sqids)
endif()

if (TARGET ch_contrib::idna)
    list (APPEND PRIVATE_LIBS ch_contrib::idna)
endif()

if (TARGET ch_contrib::h3)
    list (APPEND PRIVATE_LIBS ch_contrib::h3)
endif()

if (TARGET ch_contrib::utf8proc)
    list (APPEND PRIVATE_LIBS ch_contrib::utf8proc)
endif()

if (TARGET ch_contrib::vectorscan)
    list (APPEND PRIVATE_LIBS ch_contrib::vectorscan)
endif()

if (TARGET ch_contrib::simdjson)
    list (APPEND PRIVATE_LIBS ch_contrib::simdjson)
endif()

if (TARGET ch_contrib::rapidjson)
    list (APPEND PRIVATE_LIBS ch_contrib::rapidjson)
endif()

if (TARGET ch_contrib::pocketfft)
    list (APPEND PRIVATE_LIBS ch_contrib::pocketfft)
endif()

if (TARGET ch_contrib::crc32-vpmsum)
    list (APPEND PUBLIC_LIBS ch_contrib::crc32-vpmsum)
endif()

add_subdirectory(GatherUtils)
list (APPEND PRIVATE_LIBS clickhouse_functions_gatherutils)

add_subdirectory(URL)
list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_url>)

add_subdirectory(array)
list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_array>)

add_subdirectory(Kusto)
list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_kusto>)

if (TARGET ch_contrib::datasketches)
    add_subdirectory(UniqTheta)
    list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_uniqtheta>)
endif()

add_subdirectory(JSONPath)
list (APPEND PRIVATE_LIBS clickhouse_functions_jsonpath)

add_subdirectory(keyvaluepair)
list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_extractkeyvaluepairs>)

# Signed integer overflow on user-provided data inside boost::geometry - ignore.
set_source_files_properties("pointInPolygon.cpp" PROPERTIES COMPILE_FLAGS -fno-sanitize=signed-integer-overflow)

if (USE_GPERF)
    # Only for regenerating
    add_custom_target(generate-html-char-ref-gperf ./HTMLCharacterReference.sh
            SOURCES ./HTMLCharacterReference.sh
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/HTMLCharacterReference.gperf"
            )
    add_custom_target(generate-html-char-ref ${GPERF} -t HTMLCharacterReference.gperf --output-file=HTMLCharacterReference.generated.cpp
            && clang-format -i HTMLCharacterReference.generated.cpp
            # for clang-tidy, since string.h is deprecated
            && sed -i 's/\#include <string.h>/\#include <cstring>/g' HTMLCharacterReference.generated.cpp
            SOURCES HTMLCharacterReference.gperf
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
    add_dependencies(generate-html-char-ref generate-html-char-ref-gperf)
    if (NOT TARGET generate-source)
        add_custom_target(generate-source)
    endif ()
    add_dependencies(generate-source generate-html-char-ref)
endif ()

target_link_libraries(clickhouse_functions_obj PUBLIC ${PUBLIC_LIBS} PRIVATE ${PRIVATE_LIBS})

if (USE_STATIC_LIBRARIES OR NOT SPLIT_SHARED_LIBRARIES)
    # Used to forward the linking information to the final binaries such as clickhouse / unit_tests_dbms,
    # since such information are lost after we convert to OBJECT target
    add_library(clickhouse_functions INTERFACE)
    target_link_libraries(clickhouse_functions INTERFACE ${OBJECT_LIBS} ${PUBLIC_LIBS} ${PRIVATE_LIBS})
else()
    add_library(clickhouse_functions SHARED ${OBJECT_LIBS})
    target_link_libraries(clickhouse_functions PUBLIC ${PUBLIC_LIBS} PRIVATE ${PRIVATE_LIBS})
endif ()

if (CHDB_VERSION)
    add_compile_definitions(CHDB_VERSION_STRING="${CHDB_VERSION}")
endif ()
