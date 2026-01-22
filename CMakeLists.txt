cmake_minimum_required(VERSION 3.10)
project(ESQL)

set(CMAKE_CXX_STANDARD 17)

find_library(READLINE_LIBRARY NAMES readline)
find_path(READLINE_INCLUDE_DIR NAMES readline/readline.h)

# Check for readline availability
if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    message(STATUS "Found readline library: ${READLINE_LIBRARY}")
    message(STATUS "Found readline headers: ${READLINE_INCLUDE_DIR}")
    add_definitions(-DHAVE_READLINE)
else()
    message(WARNING "Readline not found. Using basic input mode.")
endif()

# Find LightGBM
find_library(LIGHTGBM_LIBRARY
    NAMES lightgbm liblightgbm.so liblightgbm.dylib lightgbm.lib lib_lightgbm.a
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        ${CMAKE_SOURCE_DIR}/libs
)

find_path(LIGHTGBM_INCLUDE_DIR
    NAMES LightGBM/c_api.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        ${CMAKE_SOURCE_DIR}/include
)

# Check for LightGBM availability
if(LIGHTGBM_LIBRARY AND LIGHTGBM_INCLUDE_DIR)
    message(STATUS "Found LightGBM library: ${LIGHTGBM_LIBRARY}")
    message(STATUS "Found LightGBM headers: ${LIGHTGBM_INCLUDE_DIR}")
    add_definitions(-DHAVE_LIGHTGBM)
else()
    message(WARNING "LightGBM not found. AI/ML features will be disabled.")
endif()

# Find nlohmann/json
find_package(nlohmann_json 3.7.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Downloading nlohmann/json...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Find nodesoup for Matplot++ network support
find_path(NODESOUP_INCLUDE_DIR
    NAMES nodesoup.hpp
    PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        ${CMAKE_SOURCE_DIR}/nodesoup/include
        ${CMAKE_CURRENT_BINARY_DIR}/_deps/nodesoup-src/include
)

find_library(NODESOUP_LIBRARY
    NAMES nodesoup libnodesoup.a libnodesoup.so libnodesoup.dylib
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        ${CMAKE_SOURCE_DIR}/nodesoup/build
)

# If nodesoup not found in standard locations, try to build it
if(NOT NODESOUP_INCLUDE_DIR OR NOT NODESOUP_LIBRARY)
    message(STATUS "nodesoup not found, attempting to build from source...")

    # Download and build nodesoup using FetchContent
    include(FetchContent)
    FetchContent_Declare(
        nodesoup
        GIT_REPOSITORY https://github.com/olvb/nodesoup.git
        GIT_TAG master
    )
    FetchContent_GetProperties(nodesoup)
    if(NOT nodesoup_POPULATED)
        FetchContent_Populate(nodesoup)

        # Build nodesoup as a static library
        add_library(nodesoup STATIC
            ${nodesoup_SOURCE_DIR}/src/adjacent_matrix.cpp
            ${nodesoup_SOURCE_DIR}/src/algorithms.cpp
            ${nodesoup_SOURCE_DIR}/src/fruchterman_reingold.cpp
            ${nodesoup_SOURCE_DIR}/src/kamada_kawai.cpp
            ${nodesoup_SOURCE_DIR}/src/layout.cpp
            ${nodesoup_SOURCE_DIR}/src/vector2d.cpp
        )

        target_include_directories(nodesoup PUBLIC
            ${nodesoup_SOURCE_DIR}/include
        )

        set(NODESOUP_INCLUDE_DIR ${nodesoup_SOURCE_DIR}/include)
        set(NODESOUP_LIBRARY nodesoup)

        message(STATUS "Built nodesoup from source")
    endif()
endif()

# Find and add matplotplusplus
find_path(MATPLOTPP_INCLUDE_DIR
    NAMES matplot/matplot.h
    PATHS
        ${CMAKE_SOURCE_DIR}/matplotplusplus/include
        ${CMAKE_SOURCE_DIR}/matplotplusplus/source
        /usr/include
        /usr/local/include
        /opt/local/include
)

find_library(MATPLOTPP_LIBRARY
    NAMES matplot libmatplot.a libmatplot.so libmatplot.dylib
    PATHS
        ${CMAKE_SOURCE_DIR}/matplotplusplus/build
        ${CMAKE_SOURCE_DIR}/matplotplusplus/build/Release
        ${CMAKE_SOURCE_DIR}/matplotplusplus/build/Debug
        /usr/lib
        /usr/local/lib
        /opt/local/lib
)

# If not found, try to find it in the local build directory
if(NOT MATPLOTPP_LIBRARY)
    if(EXISTS "${CMAKE_SOURCE_DIR}/matplotplusplus/build/libmatplot.a")
        set(MATPLOTPP_LIBRARY "${CMAKE_SOURCE_DIR}/matplotplusplus/build/libmatplot.a")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/matplotplusplus/build/matplot/libmatplot.a")
        set(MATPLOTPP_LIBRARY "${CMAKE_SOURCE_DIR}/matplotplusplus/build/matplot/libmatplot.a")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/matplotplusplus/build/Release/libmatplot.a")
        set(MATPLOTPP_LIBRARY "${CMAKE_SOURCE_DIR}/matplotplusplus/build/Release/libmatplot.a")
    endif()
endif()

# Check for matplotplusplus availability
if(MATPLOTPP_INCLUDE_DIR AND MATPLOTPP_LIBRARY)
    message(STATUS "Found matplotplusplus library: ${MATPLOTPP_LIBRARY}")
    message(STATUS "Found matplotplusplus headers: ${MATPLOTPP_INCLUDE_DIR}")
    add_definitions(-DHAVE_MATPLOT)

    # Also check if nodesoup is available for network features
    if(NODESOUP_INCLUDE_DIR AND NODESOUP_LIBRARY)
        message(STATUS "Found nodesoup library: ${NODESOUP_LIBRARY}")
        message(STATUS "Found nodesoup headers: ${NODESOUP_INCLUDE_DIR}")
        add_definitions(-DHAVE_NODESOUP)
    else()
        message(WARNING "nodesoup not found. Network/graph plotting features will be disabled.")
        add_definitions(-DMATPLOTPP_NETWORK_ENABLED=0)
    endif()
else()
    message(WARNING "matplotplusplus not found. Visualization features will be disabled.")
endif()

# AI sources
file(GLOB AI_SOURCES
    "src/ai_executables/*.cpp"
)

#LIGHTGBM sources
file (GLOB LIGHTGBM_SOURCES
     "src/ai_executables/lightgbm/*.cpp"
)

# Execution Engine sources
file(GLOB EXECUTION_ENGINE_SOURCES
    "src/execution_engine/*.cpp"
)

# Shell executables
file(GLOB SHELL_SOURCES
    "src/shell_executables/*.cpp"
)

#Plotter executables
file(GLOB PLOTTER_SOURCES
    "src/plotter/*.cpp"
)

#Ai ExecutionEngine_Base sorces
file(GLOB AI_EXECUTION_ENGINE_BASE_SOURCES
    "src/ai_executionengine/ai_executionengine_base/*.cpp"
)

#Ai ExecutionEngine Final sorces
file(GLOB AI_EXECUTION_ENGINE_FINAL_SOURCES
    "src/ai_executionengine/ai_executionengine_final/*.cpp"
)

set(CORE_SOURCES
    src/scanner.cpp
    src/token_utils.cpp
    src/ai_parser.cpp
    src/parser.cpp
    src/ai_grammer.cpp
    src/ai_analyzer.cpp
    src/analyzer.cpp
    src/data_analysis.cpp
    src/diskstorage.cpp
    src/data_extractor.cpp
    src/database_file.cpp
    src/write_ahead_log.cpp
    src/buffer_pool.cpp
    src/page.cpp
    src/fractal_bplus_tree.cpp
    src/database.cpp
    src/database_schema.cpp
    src/main.cpp
)

# Combine all sources
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${AI_EXECUTION_ENGINE_BASE_SOURCES}
    ${AI_EXECUTION_ENGINE_FINAL_SOURCES}
    ${EXECUTION_ENGINE_SOURCES}
    ${SHELL_SOURCES}
    ${AI_SOURCES}
    ${LIGHTGBM_SOURCES}
    ${PLOTTER_SOURCES}
)

add_executable(e_sql ${ALL_SOURCES})

target_include_directories(e_sql PRIVATE
    include
    include/shell_includes
    include/execution_engine_includes
    include/ai
    include/plotter_includes
)

# Add matplotplusplus includes if found
if(MATPLOTPP_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${MATPLOTPP_INCLUDE_DIR})
endif()

# Add nodesoup includes if found
if(NODESOUP_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${NODESOUP_INCLUDE_DIR})
endif()

# Add AI/ML includes
if(LIGHTGBM_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${LIGHTGBM_INCLUDE_DIR})
endif()

if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(e_sql ${READLINE_LIBRARY})
endif()

# Add matplotplusplus library if found
if(MATPLOTPP_LIBRARY)
    target_link_libraries(e_sql ${MATPLOTPP_LIBRARY})

    # matplotplusplus dependencies
    find_package(OpenGL REQUIRED)
    target_link_libraries(e_sql OpenGL::GL)

    # Additional dependencies for matplotplusplus
    find_package(Freetype)
    if(Freetype_FOUND)
        target_link_libraries(e_sql Freetype::Freetype)
    endif()

    # Add nodesoup library if found
    if(NODESOUP_LIBRARY)
        target_link_libraries(e_sql ${NODESOUP_LIBRARY})
    endif()
endif()

if(LIGHTGBM_LIBRARY)
    target_link_libraries(e_sql ${LIGHTGBM_LIBRARY})
endif()

# Additional libraries for LightGBM
if(LIGHTGBM_LIBRARY)
    # OpenMP for parallel processing
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(e_sql OpenMP::OpenMP_CXX)
    endif()

    # Boost libraries that LightGBM may need
    cmake_policy(SET CMP0167 OLD)
    find_package(Boost COMPONENTS filesystem system program_options)
    if(Boost_FOUND)
        target_link_libraries(e_sql ${Boost_LIBRARIES})
    endif()
endif()

# Add standard C++ libraries that might be needed
target_link_libraries(e_sql stdc++fs)  # For filesystem support (C++17)
