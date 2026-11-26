add_library(libtpu_cpp STATIC
    libtpu.cpp
)

target_include_directories(libtpu_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(libtpu_cpp PRIVATE ${STRICT_WARN_FLAGS} ${COVERAGE_FLAGS})
target_link_options(libtpu_cpp PUBLIC ${COVERAGE_FLAGS})

if(NOT DEFINED LIBTPU_RUST_STATICLIB)
    if(DEFINED ENV{CARGO_TARGET_DIR} AND EXISTS "$ENV{CARGO_TARGET_DIR}/debug/libtpu.a")
        set(LIBTPU_RUST_STATICLIB "$ENV{CARGO_TARGET_DIR}/debug/libtpu.a")
    else()
        execute_process(
            COMMAND cargo metadata --format-version 1 --no-deps
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
            OUTPUT_VARIABLE CARGO_METADATA_JSON
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "\"target_directory\":\"([^\"]+)\"" _CARGO_TGT_MATCH "${CARGO_METADATA_JSON}")
        if(CMAKE_MATCH_1 AND EXISTS "${CMAKE_MATCH_1}/debug/libtpu.a")
            set(LIBTPU_RUST_STATICLIB "${CMAKE_MATCH_1}/debug/libtpu.a")
        else()
            set(LIBTPU_RUST_STATICLIB "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug/libtpu.a")
        endif()
    endif()
endif()

target_link_libraries(libtpu_cpp PUBLIC "${LIBTPU_RUST_STATICLIB}")
