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

# AI sources
file(GLOB AI_SOURCES
    "src/ai_executables/*.cpp"
)

# Execution Engine sources
file(GLOB EXECUTION_ENGINE_SOURCES
    "src/execution_engine/*.cpp"
)

#Shell executables
file(GLOB SHELL_SOURCES
    "src/shell_executables/*.cpp"
)

set(CORE_SOURCES
    src/scanner.cpp
    src/token_utils.cpp
    src/ai_parser.cpp
    src/parser.cpp
    src/ai_analyzer.cpp
    src/analyzer.cpp
    src/ai_execution_engine.cpp
    src/data_analysis.cpp
    src/ai_execution_engine_final.cpp
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
    ${EXECUTION_ENGINE_SOURCES}
    ${SHELL_SOURCES}
    ${AI_SOURCES}
)

add_executable(e_sql ${ALL_SOURCES})

target_include_directories(e_sql PRIVATE
    include
    include/shell_includes
    include/execution_engine_includes
    include/ai
)

# Add AI/ML includes
if(LIGHTGBM_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${LIGHTGBM_INCLUDE_DIR})
endif()

if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(e_sql ${READLINE_LIBRARY})
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

    # Boost libraries that LightGBM
    cmake_policy(SET CMP0167 OLD)
    find_package(Boost COMPONENTS filesystem system program_options)
    if(Boost_FOUND)
        target_link_libraries(e_sql ${Boost_LIBRARIES})
    endif()
endif()

