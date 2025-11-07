cmake_minimum_required(VERSION 3.10)
project(Esql CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Include directories
include_directories(.)

# Source files
set(SRC_FILES
    analyzer.cpp
    buffer_pool.cpp
    database.cpp
    database_file.cpp
    database_schema.cpp
    diskstorage.cpp
    executionengine.cpp
    fractal_bplus_tree.cpp
    main.cpp
    parser.cpp
    scanner.cpp
    shell.cpp
    write_ahead_log.cpp
)

# Headers ( for IDE organization)
set(HEADER_FILES
    analyzer.h
    buffer_pool.h
    common_types.h
    database.h
    database_file.h
    diskstorage.h
    executionengine.h
    fractal_bplus_tree.h
    parser.h
    scanner.h
    shell.h
    storage.h
    write_ahead_log.h
)

# Create executable
add_executable(e_sql ${SRC_FILES} ${HEADER_FILES})

# Set output name to match your existing binary
set_target_properties(e_sql PROPERTIES OUTPUT_NAME "e_sql")
