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
    src/parser.cpp
    src/analyzer.cpp     
    src/diskstorage.cpp
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
)

add_executable(e_sql ${ALL_SOURCES})

target_include_directories(e_sql PRIVATE 
    include
    include/shell_includes
    include/execution_engine_includes
)

if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(e_sql ${READLINE_LIBRARY})
endif()

