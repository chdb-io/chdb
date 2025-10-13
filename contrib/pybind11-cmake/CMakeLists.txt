option(ENABLE_PYBIND11 "Enable pybind11" 1)
if (NOT ENABLE_PYBIND11)
    message(STATUS "Not using pybind11")
    return()
endif()

string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

if (DEFINED PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION)
    # Set Python root hint for pyenv installations
    if(DEFINED ENV{PYENV_VERSION})
        set(Python_ROOT_DIR "$ENV{PYENV_ROOT}/versions/$ENV{PYENV_VERSION}")
        message(STATUS "Using Python_ROOT_DIR: ${Python_ROOT_DIR}")
    endif()

    # First try to find with EXACT version
    find_package(Python ${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION} EXACT COMPONENTS Interpreter Development)

    if(NOT Python_FOUND)
        message(STATUS "Exact Python ${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION} not found, trying with manual hints...")

        # Try to manually set paths if pyenv is being used
        if(DEFINED ENV{PYENV_VERSION})
            set(Python_EXECUTABLE "$ENV{PYENV_ROOT}/versions/$ENV{PYENV_VERSION}/bin/python")
            set(Python_INCLUDE_DIRS "$ENV{PYENV_ROOT}/versions/$ENV{PYENV_VERSION}/include/python${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION}")
            set(Python_LIBRARY_DIRS "$ENV{PYENV_ROOT}/versions/$ENV{PYENV_VERSION}/lib")

            # Try to find the library file
            find_library(Python_LIBRARIES
                NAMES python${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION} python${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION}m
                PATHS "${Python_LIBRARY_DIRS}"
                NO_DEFAULT_PATH
            )

            if(Python_EXECUTABLE AND Python_INCLUDE_DIRS AND Python_LIBRARIES)
                set(Python_FOUND TRUE)
                set(Python_VERSION ${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION})
                string(REPLACE "." ";" VERSION_LIST ${Python_VERSION})
                list(GET VERSION_LIST 0 Python_VERSION_MAJOR)
                list(GET VERSION_LIST 1 Python_VERSION_MINOR)
                message(STATUS "Manually configured Python ${Python_VERSION}")
                message(STATUS "  Executable: ${Python_EXECUTABLE}")
                message(STATUS "  Include dirs: ${Python_INCLUDE_DIRS}")
                message(STATUS "  Libraries: ${Python_LIBRARIES}")
            endif()
        endif()

        if(NOT Python_FOUND)
            message(FATAL_ERROR "Could NOT find Python ${PYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION} development files")
        endif()
    endif()
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
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
    Py_LIMITED_API=0x03080000
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
    # For musl cross-compilation, add nostartfiles to avoid crtbeginS.o/crtendS.o issues
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

if(Python_FOUND)
    target_include_directories(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE ${PYBIND11_INCLUDE_DIR})
    target_include_directories(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE ${Python_INCLUDE_DIRS})
    target_link_libraries(${PYBIND11_NONLIMITEDAPI_LIBNAME} PUBLIC ch_contrib::pybind11_stubs)
    if (APPLE)
        target_link_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -undefined dynamic_lookup)
    else()
        target_link_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -Wl,--unresolved-symbols=ignore-all)
        # For musl cross-compilation, add nostartfiles to avoid crtbeginS.o/crtendS.o issues
        if (USE_MUSL)
            target_link_options(${PYBIND11_NONLIMITEDAPI_LIBNAME} PRIVATE -nostartfiles)
        endif()
    endif()
else()
    message(FATAL_ERROR "Python ${PY_VER} not found, aborting build")
endif()
