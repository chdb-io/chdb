add_subdirectory(divide)

include("${ClickHouse_SOURCE_DIR}/cmake/dbms_glob_sources.cmake")
add_headers_and_sources(clickhouse_functions .)

option(CHDB_MINIMAL_FUNCTIONS "Build with minimal SQL functions for chdb" OFF)

if (CHDB_MINIMAL_FUNCTIONS)
    message(STATUS "Building with MINIMAL SQL functions (whitelist mode)")
    set(clickhouse_functions_sources "")
    list(APPEND clickhouse_functions_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/IFunction.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/IFunctionAdaptors.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionDynamicAdaptor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionFactory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionHelpers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionsLogical.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/if.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/multiIf.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/multiMatchAny.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/checkHyperscanRegexp.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CastOverloadResolver.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionsConversion.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionsConversion_impl0.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionsConversion_impl1.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionsConversion_impl2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionsConversion_impl3.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/searchAnyAll.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/extractTimeZoneFromFunctionArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/generateSnowflakeID.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/registerFunctions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FunctionGenerateRandomStructure.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/chdbVersion.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/serverConstants.cpp"
        # Comparison functions
        "${CMAKE_CURRENT_SOURCE_DIR}/equals.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/notEquals.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/less.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/lessOrEquals.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/greater.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/greaterOrEquals.cpp"
    )
endif()

# These files are moved from clickhouse_functions into dbms to make dbms self-contained (not depend on clickhouse_functions)
# This allows less dependency and linker work (specially important when building many example executables)
set(DBMS_FUNCTIONS
    IFunction.cpp # IFunctionOverloadResolver::getLambdaArgumentTypes, IExecutableFunction::execute... (Many AST visitors, analyzer passes, some storages...)
    IFunctionAdaptors.cpp # FunctionToFunctionBaseAdaptor (Used by FunctionFactory.cpp)
    FunctionDynamicAdaptor.cpp # IFunctionOverloadResolver::getLambdaArgumentTypes, IExecutableFunction::execute... (Many AST visitors, analyzer passes, some storages...)
    FunctionVariantAdaptor.cpp # IFunctionOverloadResolver::build, IExecutableFunction::execute... (For Variant type support)
    FunctionFactory.cpp # FunctionFactory::instance() (Many AST visitors, analyzer passes, some storages...)
    FunctionHelpers.cpp # convertConstTupleToConstantElements, checkAndGetColumnConstStringOrFixedString, checkAndGetNestedArrayOffset ...)
    FunctionsLogical.cpp # createInternalFunctionAndOverloadResolve / createInternalFunctionOrOverloadResolver ... (Multiple)
    if.cpp # createInternalFunctionIfOverloadResolver (Used by MultiIfToIfPass.cpp)
    multiIf.cpp # createInternalMultiIfOverloadResolver (Used by IfChainToMultiIfPass.cpp)
    multiMatchAny.cpp # createInternalMultiMatchAnyOverloadResolver (Used by ConvertOrLikeChainPass.cpp)
    checkHyperscanRegexp.cpp # checkHyperscanRegexp (Used by MultiMatchAnyImpl.h, multiMatchAny.cpp)
    CastOverloadResolver.cpp # createInternalCast (Used by WindowTransform.cpp, KeyCondition.cpp...)
    FunctionsConversion.cpp # createFunctionBaseCast (Used by CastOverloadResolver.cpp)
    hasAnyAllTokens.cpp # FunctionSearchImpl (Used by MergeTreeIndexConditionText.cpp)
    FunctionsConversion_impl00.cpp
    FunctionsConversion_impl01.cpp
    FunctionsConversion_impl02.cpp
    FunctionsConversion_impl03.cpp
    FunctionsConversion_impl04.cpp
    FunctionsConversion_impl05.cpp
    FunctionsConversion_impl06.cpp
    FunctionsConversion_impl07.cpp
    FunctionsConversion_impl08.cpp
    FunctionsConversion_impl09.cpp
    FunctionsConversion_impl10.cpp
    FunctionsConversion_impl11.cpp
    FunctionsConversion_impl12.cpp
    FunctionsConversion_impl13.cpp
    FunctionsConversion_impl14.cpp
    FunctionsConversion_impl15.cpp
    FunctionsConversion_impl16.cpp
    FunctionsConversion_impl17.cpp
    FunctionsConversion_impl18.cpp
    FunctionsConversion_impl19.cpp
    FunctionsConversion_impl20.cpp
    FunctionsConversion_impl21.cpp
    FunctionsConversion_impl22.cpp
    FunctionsConversion_impl23.cpp
    FunctionsConversion_impl24.cpp
    FunctionsConversion_impl25.cpp
    FunctionsConversion_impl26.cpp
    FunctionsConversion_impl27.cpp
    FunctionsConversion_impl28.cpp
    FunctionsConversion_impl1.cpp
    FunctionsConversion_impl2.cpp
    FunctionsConversion_impl3.cpp
    FunctionTopKFilter.cpp # createInternalFunctionTopKFilterResolver used from optimizeTopK.cpp
    extractTimeZoneFromFunctionArguments.cpp # extractTimeZoneFromFunctionArguments (DateTimeTransforms.h, FunctionsConversion.cpp)
    generateSnowflakeID.cpp
)
extract_into_parent_list(clickhouse_functions_sources dbms_sources ${DBMS_FUNCTIONS})
extract_into_parent_list(clickhouse_functions_headers dbms_headers
    IFunction.h
    FunctionFactory.h
    FunctionHelpers.h
    extractTimeZoneFromFunctionArguments.h
    FunctionsLogical.h
    CastOverloadResolver.h
)

add_library(clickhouse_functions_obj OBJECT ${clickhouse_functions_headers} ${clickhouse_functions_sources})
if (OMIT_HEAVY_DEBUG_SYMBOLS)
    target_compile_options(clickhouse_functions_obj PRIVATE "-g1")
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
        boost::filesystem
        divide_impl
        ch_contrib::xxHash
        ch_contrib::stringzilla
)

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

if (TARGET ch_contrib::bech32)
    list (APPEND PRIVATE_LIBS ch_contrib::bech32)
endif()

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

add_subdirectory(TimeSeries)
list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_time_series>)

if (TARGET ch_contrib::datasketches)
    add_subdirectory(UniqTheta)
    list (APPEND OBJECT_LIBS $<TARGET_OBJECTS:clickhouse_functions_uniqtheta>)
endif()

add_subdirectory(JSONPath)
list (APPEND PRIVATE_LIBS clickhouse_functions_jsonpath)

add_subdirectory(keyvaluepair)

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

# Update after all the includes
set(dbms_sources "${dbms_sources}" PARENT_SCOPE)
set(dbms_headers "${dbms_headers}" PARENT_SCOPE)

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
