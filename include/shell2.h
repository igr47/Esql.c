#ifndef ESQL_SHELL_H
#define ESQL_SHELL_H

#include "database.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <termios.h>
#include <cstddef>

class ESQLShell {
public:
    explicit ESQLShell(Database& db);
    ~ESQLShell();
    
    void run();
    void setCurrentDatabase(const std::string& db_name);

private:
    // Platform detection
    enum class Platform { Linux, Windows, Termux, Unknown };

    struct Cell {
        std::string bytes;      // UTF-8 bytes for this grapheme
        int width;              // visual width as returned by wcwidth
        std::string prefix;     // ANSI sequences that should be printed BEFORE this cell
    };

    struct VisualLine {
        std::string rendered; // entire rendered line (used for quick debug / output)
        std::vector<Cell> cells;
    };

    struct CursorPos {
        int row;
        int col;
    };

    struct RenderCache {
        std::vector<Cell> cells;
        std::vector<std::vector<Cell>> wrapped_rows;
        int cached_terminal_width = 0;
        int cached_prompt_width = 0;
        std::string input_hash;
        bool valid = false;
    };

    // New re-render engine
    void rebuild_cells_from_input(const std::string& input);
    void attach_ansi_prefixes_from_colorized(const std::string& colorized);
    std::vector<std::vector<Cell>> wrap_cells_for_prompt(int prompt_width);
    CursorPos cursor_position_from_byte_offset(size_t byte_offset, int prompt_width);
    void render_input_full();
    size_t byte_offset_at_cell_index(size_t cell_index) const;
    
    // Caching system
    void invalidate_cache();
    bool use_cached_render(int prompt_width);

    // Existing API (kept names)
    //LineWrapInfo calculate_enhanced_wrapping(const std::string& input, size_t cursor_pos); // UNUSED but kept for compatibility
    int last_displayed_lines = 0;
    void redraw_input_enhanced(); // wrapper to new render_input_full
    void redraw_input();
    void refresh_display();
    void handle_terminal_resize();

    Platform detect_platform();
    void process_word(const std::string& word, std::string& result,const std::unordered_set<std::string>& aggregate_functions,const std::unordered_set<std::string>& operators);

    // Terminal control
    void enable_raw_mode();
    void disable_raw_mode();
    void get_terminal_size();
    void clear_screen();
    
    // Input handling
    int read_key();
    void handle_enter();
    void insert_char(char c);
    void delete_char();
    void handle_tab_completion();
    
    // Navigation
    void move_cursor_left();
    void move_cursor_right();
    void navigate_history_up();
    void navigate_history_down();

    // Utility helpers
    std::string strip_ansi_simple(const std::string& s);
    std::vector<Cell> utf8_to_base_cells(const std::string& s);
    int display_width(const std::string& s);

    // Display and rendering
    void print_banner();
    std::vector<std::string> load_ascii_art();
    std::string color_line(const std::string& line, const char* color);
    void print_prompt();
    std::string colorize_sql(const std::string& input);
    void print_results(const ExecutionEngine::ResultSet& result, double duration);
    void print_structure_results(const ExecutionEngine::ResultSet& result, double duration);
    void show_help();
    
    // History and completion
    void add_to_history(const std::string& command);
    std::vector<std::string> get_completion_suggestion(const std::string& input);
    
    // Utility
    std::string get_current_time() const;
    bool is_single_line_command(const std::string& command) const;
    void execute_command(const std::string& command);
    
    // Platform-specific runners
    void run_termux();
    void run_linux();
    bool is_termux() const;
    
    Database& db;
    std::string current_db;
    std::string current_line;    // raw user input bytes (UTF-8)
    size_t cursor_pos = 0;       // byte offset into current_line

    // Cell buffer rebuilt from current_line
    std::vector<Cell> cell_buffer;

    // Render cache for performance
    RenderCache render_cache;
    
    std::vector<std::string> command_history;
    int history_index = -1;
    
    int terminal_width = 80;
    int banner_lines = 0; // computed at runtime for render positioning
    
    Platform current_platform = Platform::Unknown;
    bool use_colors = true;
    
    #ifdef _WIN32
    HANDLE hInput;
    HANDLE hOutput;
    DWORD original_mode;
    #else
    struct termios orig_termios;
    #endif
    
    // ANSI color codes with bold support
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
    
    // SQL elements for syntax highlighting and completion
    static const std::unordered_set<std::string> keywords;
    static const std::unordered_set<std::string> datatypes;
    static const std::unordered_set<std::string> constraints;
    static const std::unordered_set<std::string> conditionals;

    // Prompt format (prompt depends on current_db and time)
    std::string build_prompt() const;
};

#endif // ESQL_SHELL_H
