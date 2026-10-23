cmake_minimum_required(VERSION 3.20)
project(libtpu_cpp VERSION 0.0.50 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_COVERAGE "Enable gcov/llvm-cov coverage instrumentation" ON)

set(STRICT_WARN_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wnull-dereference
    -Wformat=2
    -fno-exceptions
    -fno-rtti
)

if(ENABLE_COVERAGE)
    set(COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping -O0 -g)
else()
    set(COVERAGE_FLAGS "")
endif()

add_subdirectory(src)

enable_testing()
add_executable(test_libtpu tests/test_libtpu.cpp)
target_compile_options(test_libtpu PRIVATE ${STRICT_WARN_FLAGS} ${COVERAGE_FLAGS})
target_link_options(test_libtpu PRIVATE ${COVERAGE_FLAGS})
target_link_libraries(test_libtpu PRIVATE libtpu_cpp)
add_test(NAME test_libtpu COMMAND test_libtpu)
