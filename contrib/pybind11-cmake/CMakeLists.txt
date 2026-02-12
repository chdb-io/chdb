option(ENABLE_PYBIND11 "Enable pybind11" 1)
if (NOT ENABLE_PYBIND11)
    message(STATUS "Not using pybind11")
    return()
endif()

string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

# Use custom Python path if provided
if (DEFINED CHDB_CUSTOM_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE "${CHDB_CUSTOM_PYTHON_EXECUTABLE}")
    set(Python3_EXECUTABLE "${CHDB_CUSTOM_PYTHON_EXECUTABLE}")
    message(STATUS "Using custom Python executable: ${CHDB_CUSTOM_PYTHON_EXECUTABLE}")
endif()

if(CHDB_CROSSCOMPILING)
    # For cross-compiling, manually set Python variables
    if (DEFINED PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION)
        # Parse version like "3.8" into major and minor
        string(REPLACE "." ";" VERSION_LIST ${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION})
        list(GET VERSION_LIST 0 Python_VERSION_MAJOR)
        list(GET VERSION_LIST 1 Python_VERSION_MINOR)
        set(Python_VERSION "${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION}")

        # Set include directory based on provided prefix and version
        set(Python_INCLUDE_DIRS "${CHDB_PYTHON_INCLUDE_DIR_PREFIX}/${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
    else()
        # Default to Python 3.9 if not specified
        set(Python_VERSION_MAJOR 3)
        set(Python_VERSION_MINOR 9)
        set(Python_VERSION "3.9")
        set(Python_INCLUDE_DIRS "${CHDB_PYTHON_INCLUDE_DIR_PREFIX}/3.9")
    endif()

    message(STATUS "Cross-compiling: Using Python ${Python_VERSION} include directory: ${Python_INCLUDE_DIRS}")
else()
    if (DEFINED PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION)
        find_package(Python ${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development)
    else()
        find_package(Python REQUIRED COMPONENTS Interpreter Development)
    endif()
endif()

set(LIBRARY_DIR "${ClickHouse_SOURCE_DIR}/contrib/pybind11/")
set(PYBIND11_INCLUDE_DIR "${LIBRARY_DIR}/include" CACHE STRING "Path to pybind11 include directory")
message(STATUS "Using stable-ABI pybind11 from ${LIBRARY_DIR}")

set(PYBIND11_LIB_SUFFIX "chdb")

# Build the main pybind11nonlimitedapi_stubs library
set(STUBS_SOURCES
    "${LIBRARY_DIR}/source/non_limited_api/non_limited_api_stubs.cpp"
)

if (NOT USE_MUSL)
    add_library(pybind11_stubs SHARED ${STUBS_SOURCES})
else()
    add_library(pybind11_stubs STATIC ${STUBS_SOURCES})
endif()
target_include_directories(pybind11_stubs PRIVATE ${PYBIND11_INCLUDE_DIR})
target_include_directories(pybind11_stubs PRIVATE ${Python_INCLUDE_DIRS})
target_compile_definitions(pybind11_stubs PUBLIC
    PYBIND11_NONLIMITEDAPI_LIB_SUFFIX_FOR_MODULE="${PYBIND11_LIB_SUFFIX}"
    Py_LIMITED_API=0x03090000
)

if (OS_LINUX AND Python_VERSION_MAJOR EQUAL 3 AND Python_VERSION_MINOR EQUAL 8)
    target_compile_options(pybind11_stubs PRIVATE -w -idirafter /usr/include -include crypt.h)
endif()

set_target_properties(pybind11_stubs PROPERTIES
    OUTPUT_NAME "pybind11nonlimitedapi_stubs"
)

if (APPLE)
    target_link_options(pybind11_stubs PRIVATE -undefined dynamic_lookup)
else()
    target_link_options(pybind11_stubs PRIVATE -Wl,--unresolved-symbols=ignore-all)
    if (USE_MUSL)
        target_link_options(pybind11_stubs PRIVATE -nostartfiles)
    endif()
endif()

add_library(ch_contrib::pybind11_stubs ALIAS pybind11_stubs)

set(PYBIND11_NONLIMITEDAPI_LIBNAME
    "pybind11nonlimitedapi_${PYBIND11_LIB_SUFFIX}_${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")

add_library(${PYBIND11_NONLIMITEDAPI_LIBNAME} SHARED ${LIBRARY_DIR}/source/non_limited_api/non_limited_api.cpp)

if (OS_LINUX AND Python_VERSION_MAJOR EQUAL 3 AND Python_VERSION_MINOR EQUAL 8)
    target_compile_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -w -idirafter /usr/include -include crypt.h)
endif()

set_target_properties(${PYBIND11_NONLIMITEDAPI_LIBNAME} PROPERTIES
    OUTPUT_NAME "${PYBIND11_NONLIMITEDAPI_LIBNAME}"
)

message(STATUS "Python_VERSION: ${Python_VERSION}")
message(STATUS "Python_VERSION_MAJOR: ${Python_VERSION_MAJOR}")
message(STATUS "Python_VERSION_MINOR: ${Python_VERSION_MINOR}")
message(STATUS "Resulting LIBNAME: ${PYBIND11_NONLIMITEDAPI_LIBNAME}")

if(Python_FOUND OR CHDB_CROSSCOMPILING)
    target_include_directories(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE ${PYBIND11_INCLUDE_DIR})
    target_include_directories(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE ${Python_INCLUDE_DIRS})
    target_link_libraries(${PYBIND11_NONLIMITEDAPI_LIBNAME} PUBLIC ch_contrib::pybind11_stubs)

    if (USE_MUSL AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        target_link_libraries(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -Wl,--whole-archive -lgcc -Wl,--no-whole-archive)
        message(STATUS "Linking whole libgcc archive for ARM64+musl to provide all compiler runtime symbols")
    endif()

    if (APPLE)
        target_link_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -undefined dynamic_lookup)
    else()
        target_link_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -Wl,--unresolved-symbols=ignore-all)
        if (USE_MUSL)
            target_link_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -nostartfiles)
        endif()
    endif()
else()
    message(FATAL_ERROR "Python ${PY_VER} not found, aborting build")
endif()
