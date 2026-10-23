add_library(libtpu_cpp STATIC
    libtpu.cpp
)

target_include_directories(libtpu_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(libtpu_cpp PRIVATE ${STRICT_WARN_FLAGS} ${COVERAGE_FLAGS})
target_link_options(libtpu_cpp PUBLIC ${COVERAGE_FLAGS})

if(NOT DEFINED LIBTPU_RUST_STATICLIB)
    set(LIBTPU_RUST_STATICLIB "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug/libtpu.a")
endif()

target_link_libraries(libtpu_cpp PUBLIC "${LIBTPU_RUST_STATICLIB}")
