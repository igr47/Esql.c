#ifndef SHELL_TYPES_H
#define SHELL_TYPES_H

#include <string>
#include <vector>
#include <cstddef>

namespace esql {

// UTF-8 character cell
struct Cell {
    std::string bytes;      // UTF-8 bytes for this grapheme
    int width;              // visual width as returned by wcwidth
    std::string prefix;     // ANSI sequences that should be printed BEFORE this cell
};

// Cursor position in wrapped coordinates
struct CursorPos {
    int row;
    int col;
};

// Wrapped line information
struct WrappedLine {
    std::vector<Cell> cells;
    size_t byte_start = 0;
    size_t byte_end = 0;
    bool is_continuation = false;
};

// Render cache for performance
struct RenderCache {
    std::vector<WrappedLine> wrapped_lines;
    int terminal_width = 0;
    int prompt_width = 0;
    std::string input_hash;
    bool valid = false;
};

struct AutoSuggestion {
    std::string suggestion;
    std::string prefix;
    bool active = false;
    size_t display_start = 0;
};

// Key codes for cross-platform input
enum class KeyCode {
    None = -1,
    Character = 0,
    Up = 1002,
    Down = 1003,
    Right = 1001,
    Left = 1000,
    Enter = '\n',
    Tab = '\t',
    Backspace = 127,
    Delete = 8,
    Escape = '\033',
    CtrlD = 4,
    CtrlR = 18,
    CtrlL = 12,
    Home = 1004,
    End = 1005
};

// ANSI color codes
namespace colors {
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* BOLD = "\033[1m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";
    static constexpr const char* GRAY = "\033[90m";
    static constexpr const char* BOLD_BLUE = "\033[1;34m";
    static constexpr const char* BOLD_CYAN = "\033[1;36m";
    static constexpr const char* BOLD_GREEN = "\033[1;32m";
    static constexpr const char* BOLD_MAGENTA = "\033[1;35m";
    static constexpr const char* BOLD_RED = "\033[1;31m";
}

} // namespace esql

#endif
