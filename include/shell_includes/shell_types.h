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
    static constexpr const char* BRIGHT_RED = "\033[91m";
    static constexpr const char* TEAL = "\033[38;5;43m";
    static constexpr const char* LAVENDER = "\033[38;5;183m";
    static constexpr const char* MINT = "\033[38;5;85m";
    static constexpr const char* CORAL = "\033[38;5;209m";
    static constexpr const char* GOLD = "\033[38;5;220m"; 
    
    static constexpr const char* DIM = "\033[2m";
    static constexpr const char* ITALIC = "\033[3m";
    static constexpr const char* UNDERLINE = "\033[4m";

    static constexpr const char* BRIGHT_GREEN = "\033[92m";
    static constexpr const char* BRIGHT_YELLOW = "\033[93m";
    static constexpr const char* BRIGHT_BLUE = "\033[94m";
    static constexpr const char* BRIGHT_MAGENTA = "\033[95m";
    static constexpr const char* BRIGHT_CYAN = "\033[96m";
    static constexpr const char* BRIGHT_WHITE = "\033[97m";

        // Professional gradient palettes
    static constexpr const char* GRADIENT_BLUE_1 = "\033[38;5;39m";
    static constexpr const char* GRADIENT_BLUE_2 = "\033[38;5;45m";
    static constexpr const char* GRADIENT_BLUE_3 = "\033[38;5;51m";
    static constexpr const char* GRADIENT_BLUE_4 = "\033[38;5;87m";

    static constexpr const char* GRADIENT_PURPLE_1 = "\033[38;5;99m";
    static constexpr const char* GRADIENT_PURPLE_2 = "\033[38;5;105m";
    static constexpr const char* GRADIENT_PURPLE_3 = "\033[38;5;111m";
    static constexpr const char* GRADIENT_PURPLE_4 = "\033[38;5;117m";
    static constexpr const char* GRADIENT_CYAN_1 = "\033[38;5;43m";
    static constexpr const char* GRADIENT_CYAN_2 = "\033[38;5;49m";
    static constexpr const char* GRADIENT_CYAN_3 = "\033[38;5;50m";
    static constexpr const char* GRADIENT_CYAN_4 = "\033[38;5;51m";

    static constexpr const char* GRADIENT_GREEN_1 = "\033[38;5;46m";
    static constexpr const char* GRADIENT_GREEN_2 = "\033[38;5;47m";
    static constexpr const char* GRADIENT_GREEN_3 = "\033[38;5;48m";
    static constexpr const char* GRADIENT_GREEN_4 = "\033[38;5;49m";
    static constexpr const char* GRADIENT_ORANGE_1 = "\033[38;5;202m";
    static constexpr const char* GRADIENT_ORANGE_2 = "\033[38;5;208m";
    static constexpr const char* GRADIENT_ORANGE_3 = "\033[38;5;214m";
    static constexpr const char* GRADIENT_ORANGE_4 = "\033[38;5;220m";

    static constexpr const char* GRADIENT_MAGENTA_1 = "\033[38;5;127m";
    static constexpr const char* GRADIENT_MAGENTA_2 = "\033[38;5;129m";
    static constexpr const char* GRADIENT_MAGENTA_3 = "\033[38;5;165m";
    static constexpr const char* GRADIENT_MAGENTA_4 = "\033[38;5;201m";

        // Premium accent colors
    static constexpr const char* PREMIUM_GOLD = "\033[38;5;220m";
    static constexpr const char* PREMIUM_SILVER = "\033[38;5;255m";
    static constexpr const char* PREMIUM_EMERALD = "\033[38;5;46m";
    static constexpr const char* PREMIUM_SAPPHIRE = "\033[38;5;27m";
    static constexpr const char* PREMIUM_RUBY = "\033[38;5;196m";

}

} // namespace esql

#endif
