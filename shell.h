#ifndef ESQLSHELL_H
#define ESQLSHELL_H

#include <string>
#include <vector>
#include <map>
#include <stack>
#include <termios.h>
#include <unordered_map>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <readline/readline.h>
#include <readline/history.h>
#include <unistd.h>
#include <set>
#include <sys/ioctl.h>
#endif

class Database;


class ESQLShell {
public:
    enum class Platform { Linux, Windows, Termux, Unknown };

    ESQLShell(Database& db);
    ~ESQLShell();
    void run();
    void setCurrentDatabase(const std::string& db_name);
    void setConnectionStatus(const std::string& status);

private:
    Database& db;
    std::string current_db;
    std::string connection_status = "disconnected";
    std::vector<std::string> command_history;
    std::stack<std::string> query_stack;
    std::stsck<std::string> query_stack2;
    static const std::unordered_set<std::string> keywords;
    static const std::unordered_set<std::string> types;
    static const std::unordered_set<std::string> conditionals;
    std::string current_line;
    size_t cursor_pos;
    int history_pos;
    int screen_rows,screen_cols;
    bool use_colors = true;

    struct termios orig_termios;
    //bool raw_mode = false;
    Platform current_platform = Platform::Unknown;

    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string GRAY = "\033[90m";

    //Handle platform setup
    Platform detect_platform();
    void platform_specific_init();
    void enable_raw_mode();
    void disable_raw_mode();
    bool setup_windows_terminal();
    void restore_windows_terminal();

    //Handle special key presses
    void get_window_size();
    void move_cursor_left();
    void move_cursor_right();
    void move_cursor_up();
    void move_cursor_down();
    void refresh_line();
    void previous_command_up();
    void previous_command_down();

    //Handle charachter handling
    void insert_char(char c);
    void delete_char();
    void redraw_line();
    void clear_line();

    //Take care of display
    void print_banner();
    void update_connection_status();
    void print_prompt();
    std::string colorize_sql();
    void show_help();
    std::string expand_alias(const std::string& input);
    void print_result_table(const std::vector<std::vector<std::string>>& rows,
                           const std::vector<std::string>& headers,
                           long long duration_ms = 0);
    void execute_command(const std::string& raw_cmd);
    void handle_special_keys();
    void handle_windows_input();
    void handle_unix_input();
    void redraw_line();
    size_t calculate_visible_position(const std::string& str, size_t logical_pos);
    void run_interactive();
};

#endif
