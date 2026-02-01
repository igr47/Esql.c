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
    static constexpr const char* BLACK = "\033[38;5;0m";
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
    static constexpr const char* BLINK = "\033[5m";
    static constexpr const char* REVERSE = "\033[7m";
    static constexpr const char* HIDDEN = "\033[8m";
    static constexpr const char* STRIKETHROUGH = "\033[9m";

    static constexpr const char* BRIGHT_GREEN = "\033[92m";
    static constexpr const char* BRIGHT_BLACK = "\033[38;5;8m";
    static constexpr const char* BRIGHT_YELLOW = "\033[93m";
    static constexpr const char* BRIGHT_BLUE = "\033[94m";
    static constexpr const char* BRIGHT_MAGENTA = "\033[95m";
    static constexpr const char* BRIGHT_CYAN = "\033[96m";
    static constexpr const char* BRIGHT_WHITE = "\033[97m";

    static constexpr const char* LIGHT_GRAY = "\033[38;5;7m";
    static constexpr const char* DARK_GRAY = "\033[38;5;236m";

    static constexpr const char* DARK_RED = "\033[38;5;88m";
    static constexpr const char* DARK_GREEN = "\033[38;5;22m";
    static constexpr const char* DARK_YELLOW = "\033[38;5;94m";
    static constexpr const char* DARK_BLUE = "\033[38;5;18m";
    static constexpr const char* DARK_MAGENTA = "\033[38;5;90m";
    static constexpr const char* DARK_CYAN = "\033[38;5;30m";

    static constexpr const char* LIGHT_GREEN = "\033[1;32m";
    static constexpr const char* LIGHT_CYAN = "\033[1;36m";

    static constexpr const char* ORANGE = "\033[38;5;208m";
    static constexpr const char* LIGHT_ORANGE = "\033[38;5;214m";
    static constexpr const char* DARK_ORANGE = "\033[38;5;166m";

    static constexpr const char* PINK = "\033[38;5;213m";
    static constexpr const char* LIGHT_PINK = "\033[38;5;218m";
    static constexpr const char* DARK_PINK = "\033[38;5;125m";

    static constexpr const char* PURPLE = "\033[38;5;93m";
    static constexpr const char* LIGHT_PURPLE = "\033[38;5;141m";
    static constexpr const char* DARK_PURPLE = "\033[38;5;55m";

    static constexpr const char* BROWN = "\033[38;5;94m";
    static constexpr const char* LIGHT_BROWN = "\033[38;5;137m";
    static constexpr const char* DARK_BROWN = "\033[38;5;52m";

    static constexpr const char* LIGHT_TEAL = "\033[38;5;14m";
    static constexpr const char* DARK_TEAL = "\033[38;5;23m";

    static constexpr const char* SILVER = "\033[38;5;7m";
    static constexpr const char* BRONZE = "\033[38;5;130m";

    static constexpr const char* FOREST_GREEN = "\033[38;5;28m";
    static constexpr const char* OCEAN_BLUE = "\033[38;5;27m";
    static constexpr const char* SKY_BLUE = "\033[38;5;39m";
    static constexpr const char* GRASS_GREEN = "\033[38;5;46m";
    static constexpr const char* SUN_YELLOW = "\033[38;5;226m";
    static constexpr const char* FIRE_RED = "\033[38;5;196m";
    static constexpr const char* ICE_BLUE = "\033[38;5;45m";
    static constexpr const char* EARTH_BROWN = "\033[38;5;94m";
    static constexpr const char* MIDNIGHT_BLUE = "\033[38;5;17m";

    // True Color RGB macros for advanced users
    #define RGB(r, g, b) "\033[38;2;" #r ";" #g ";" #b "m"
    #define RGBA(r, g, b, a) "\033[38;2;" #r ";" #g ";" #b "m" // Note: ANSI doesn't support alpha
    
    static constexpr const char* BG_BLACK = "\033[48;5;0m";
    static constexpr const char* BG_WHITE = "\033[48;5;15m";
    static constexpr const char* BG_RED = "\033[48;5;1m";
    static constexpr const char* BG_GREEN = "\033[48;5;2m";
    static constexpr const char* BG_BLUE = "\033[48;5;4m";
    static constexpr const char* BG_YELLOW = "\033[48;5;3m";
    static constexpr const char* BG_MAGENTA = "\033[48;5;5m";
    static constexpr const char* BG_CYAN = "\033[48;5;6m";
    static constexpr const char* BG_GRAY = "\033[48;5;8m";

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

    static constexpr const char* RESET_COLOR = "\033[39m";
    static constexpr const char* RESET_BG = "\033[49m";
    static constexpr const char* RESET_BOLD = "\033[22m";
    static constexpr const char* RESET_ITALIC = "\033[23m";
    static constexpr const char* RESET_UNDERLINE = "\033[24m";
    static constexpr const char* RESET_BLINK = "\033[25m";
    static constexpr const char* RESET_REVERSE = "\033[27m";
    static constexpr const char* RESET_ALL = "\033[0m";

    // Premium accent colors
    static constexpr const char* PREMIUM_GOLD = "\033[38;5;220m";
    static constexpr const char* PREMIUM_SILVER = "\033[38;5;255m";
    static constexpr const char* PREMIUM_EMERALD = "\033[38;5;46m";
    static constexpr const char* PREMIUM_SAPPHIRE = "\033[38;5;27m";
    static constexpr const char* PREMIUM_RUBY = "\033[38;5;196m";

    // Solarized colors
    static constexpr const char* BASE00 = "\033[38;5;244m";
    static constexpr const char* BASE01 = "\033[38;5;240m";
    static constexpr const char* BASE0 = "\033[38;5;250m";
    static constexpr const char* BASE1 = "\033[38;5;251m";
    static constexpr const char* BASE2 = "\033[48;5;0m";
    static constexpr const char* BASE3 = "\033[48;5;8m";
    static constexpr const char* SOLAR_VIOLET = "\033[38;5;61m";

    // Nord colors
    static constexpr const char* NORD_BLUE = "\033[38;5;109m";
    static constexpr const char* NORD_CYAN = "\033[38;5;116m";
    static constexpr const char* NORD_GREEN = "\033[38;5;114m";
    static constexpr const char* NORD_YELLOW = "\033[38;5;223m";
    static constexpr const char* NORD_ORANGE = "\033[38;5;209m";
    static constexpr const char* NORD_RED = "\033[38;5;174m";
    static constexpr const char* NORD_PURPLE = "\033[38;5;140m";
    static constexpr const char* NORD_MAGENTA = "\033[38;5;176m";
    static constexpr const char* NORD3 = "\033[38;5;240m";
    static constexpr const char* NORD4 = "\033[38;5;250m";

    // GitHub Dark colors
    static constexpr const char* GH_BLUE = "\033[38;5;68m";
    static constexpr const char* GH_CYAN = "\033[38;5;116m";
    static constexpr const char* GH_GREEN = "\033[38;5;114m";
    static constexpr const char* GH_YELLOW = "\033[38;5;185m";
    static constexpr const char* GH_ORANGE = "\033[38;5;208m";
    static constexpr const char* GH_RED = "\033[38;5;203m";
    static constexpr const char* GH_PINK = "\033[38;5;211m";
    static constexpr const char* GH_PURPLE = "\033[38;5;140m";
    static constexpr const char* GH_GRAY = "\033[38;5;246m";
    static constexpr const char* GH_FG = "\033[38;5;250m";

}

} // namespace esql

#endif
