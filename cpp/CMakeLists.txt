cmake_minimum_required(VERSION 3.20)
project(libtpu_cpp VERSION 0.0.53 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_COVERAGE "Enable gcov/llvm-cov coverage instrumentation" ON)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(STRICT_C_WARN_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wnull-dereference
    -Wformat=2
)

set(STRICT_WARN_FLAGS
    ${STRICT_C_WARN_FLAGS}
    -fno-exceptions
    -fno-rtti
)

if(ENABLE_SANITIZERS)
    set(SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
else()
    set(SANITIZER_FLAGS "")
endif()

if(ENABLE_COVERAGE)
    set(COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping -O0 -g ${SANITIZER_FLAGS})
else()
    set(COVERAGE_FLAGS ${SANITIZER_FLAGS})
endif()

add_subdirectory(src)

enable_testing()
add_executable(test_libtpu tests/test_libtpu.cpp)
target_compile_definitions(
    test_libtpu
    PRIVATE
    LIBTPU_GOLDEN_PROTO_DESC_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../docs/reverse_engineering/golden_proto_descriptors.json"
)
target_compile_options(test_libtpu PRIVATE ${STRICT_WARN_FLAGS} ${COVERAGE_FLAGS})
target_link_options(test_libtpu PRIVATE ${COVERAGE_FLAGS})
target_link_libraries(test_libtpu PRIVATE libtpu_cpp)
add_test(NAME test_libtpu COMMAND test_libtpu)

add_executable(test_libtpu_c11 tests/test_libtpu_c11.c)
target_include_directories(test_libtpu_c11 PRIVATE src)
target_compile_options(test_libtpu_c11 PRIVATE ${STRICT_C_WARN_FLAGS} ${COVERAGE_FLAGS})
target_link_options(test_libtpu_c11 PRIVATE ${COVERAGE_FLAGS})
target_link_libraries(test_libtpu_c11 PRIVATE libtpu_cpp)
add_test(NAME test_libtpu_c11 COMMAND test_libtpu_c11)
