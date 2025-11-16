// shell.h
#ifndef ESQL_SHELL_H
#define ESQL_SHELL_H

#include "database.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <termios.h>

class ESQLShell {
public:
    explicit ESQLShell(Database& db);
    ~ESQLShell();
    
    void run();
    void setCurrentDatabase(const std::string& db_name);

private:
    // Platform detection
    enum class Platform { Linux, Windows, Termux, Unknown };
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
    
    // Display and rendering
    void print_banner();
    std::vector<std::string> load_ascii_art();
     std::string color_line(const std::string& line, const char* color);
    void print_prompt();
    void redraw_interface();
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
    std::string current_line;
    size_t cursor_pos = 0;
    
    std::vector<std::string> command_history;
    int history_index = -1;
    
    int terminal_width = 80;
    
    Platform current_platform = Platform::Unknown;
    bool use_colors = true;
    
    #ifdef _WIN32
    HANDLE hInput;
    HANDLE hOutput;
    DWORD original_mode;
    #else
    struct termios orig_termios;
    #endif
    
    // ANSI color codes
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";
    static constexpr const char* GRAY = "\033[90m";
    
    // SQL elements for syntax highlighting and completion
    static const std::unordered_set<std::string> keywords;
    static const std::unordered_set<std::string> datatypes;
    static const std::unordered_set<std::string> conditionals;
};

#endif // ESQL_SHELL_H
