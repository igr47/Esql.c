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


add_executable(e_sql
    src/scanner.cpp
    src/parser.cpp
    src/analyzer.cpp     
    src/executionengine.cpp
    src/diskstorage.cpp
    src/database_file.cpp      
    src/write_ahead_log.cpp  
    src/buffer_pool.cpp
    src/page.cpp  
    src/fractal_bplus_tree.cpp
    src/database.cpp    
    src/terminal_input.cpp
    src/utf8_processor.cpp 
    src/line_renderer.cpp 
    src/syntax_highlighter.cpp
    src/hisory_manager.cpp 
    src/animator.cpp
    src/completion_engine.cpp
    src/autosuggestion_manager.cpp
    src/modern_shell.cpp 
    src/database_schema.cpp
    src/main.cpp  
)

target_include_directories(e_sql PRIVATE include)

if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_include_directories(e_sql PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(e_sql ${READLINE_LIBRARY})
endif()

