// shell.cpp
#include "shell.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <thread>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#endif

// SQL elements for syntax highlighting
const std::unordered_set<std::string> ESQLShell::keywords = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DATABASE", "DROP", "ALTER", "ADD", "RENAME",
    "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP", "DISTINCT", "DATABASES","BY","ORDER","GROUP","HAVING",
};

const std::unordered_set<std::string> ESQLShell::datatypes = {
    "INT", "INTEGER", "FLOAT", "TEXT", "STRING", "BOOL", "BOLEAN", "VARCHAR", "DATE"
};

const std::unordered_set<std::string> ESQLShell::conditionals = {
    "AND", "OR", "NOT", "NULL", "IS", "LIKE", "IN", "BETWEEN", "OFFSET", "LIMIT","AS",
};

ESQLShell::ESQLShell(Database& db) : db(db), current_db("default") {
    current_platform = detect_platform();
    get_terminal_size();
    
    if (!is_termux()) {
        enable_raw_mode();
    }
}

ESQLShell::~ESQLShell() {
    if (!is_termux()) {
        disable_raw_mode();
    }
    std::cout << RESET << std::endl;
}

ESQLShell::Platform ESQLShell::detect_platform() {
    #ifdef _WIN32
    return Platform::Windows;
    #else
    // Check for Termux
    const char* term = getenv("TERM");
    const char* pkg = getenv("PREFIX");
    const char* android_root = getenv("ANDROID_ROOT");
    
    if ((pkg && std::string(pkg).find("com.termux") != std::string::npos) ||
        (android_root) ||
        access("/data/data/com.termux/files/usr/bin/login", F_OK) == 0 ||
        access("/data/data/com.termux/files/home", F_OK) == 0) {
        return Platform::Termux;
    }
    
    return Platform::Linux;
    #endif
}

void ESQLShell::enable_raw_mode() {
    #ifdef _WIN32
    hInput = GetStdHandle(STD_INPUT_HANDLE);
    hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleMode(hInput, &original_mode);
    SetConsoleMode(hInput, original_mode & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
    #else
    tcgetattr(STDIN_FILENO, &orig_termios);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    #endif
}

void ESQLShell::disable_raw_mode() {
    #ifdef _WIN32
    SetConsoleMode(hInput, original_mode);
    #else
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    #endif
}

void ESQLShell::get_terminal_size() {
    #ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    terminal_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    #else
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    if (ws.ws_col > 0) terminal_width = ws.ws_col;
    #endif
}

void ESQLShell::clear_screen() {
    std::cout << "\033[2J\033[H";
}

int ESQLShell::read_key() {
    #ifdef _WIN32
    if (_kbhit()) {
        int ch = _getch();
        if (ch == 0 || ch == 224) { // Arrow keys
            switch (_getch()) {
                case 72: return 1002; // Up
                case 80: return 1003; // Down
                case 75: return 1000; // Left
                case 77: return 1001; // Right
            }
        }
        return ch;
    }
    return -1;
    #else
    char c;
    if (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == '\033') {
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1) {
                if (seq[0] == '[') {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1) {
                        switch (seq[1]) {
                            case 'A': return 1002; // Up
                            case 'B': return 1003; // Down
                            case 'C': return 1001; // Right
                            case 'D': return 1000; // Left
                        }
                    }
                }
            }
            return '\033';
        }
        return c;
    }
    return -1;
    #endif
}

void ESQLShell::run() {
    print_banner();
    
    if (is_termux()) {
        run_termux();
    } else {
        run_linux();
    }
}

void ESQLShell::run_linux() {
    while (true) {
        print_prompt();
        redraw_interface();
        
        int ch = read_key();
        if (ch == -1) continue;
        
        switch (ch) {
            case '\n': case '\r':
                handle_enter();
                break;
                
            case 127: case 8: // Backspace
                delete_char();
                break;
                
            case '\t': // Tab
                handle_tab_completion();
                break;
                
            case 1000: // Left arrow
                move_cursor_left();
                break;
                
            case 1001: // Right arrow
                move_cursor_right();
                break;
                
            case 1002: // Up arrow
                navigate_history_up();
                break;
                
            case 1003: // Down arrow
                navigate_history_down();
                break;
                
            case 4: // Ctrl+D
                std::cout << "\n";
                return;
                
            default:
                if (ch >= 32 && ch <= 126) {
                    insert_char(static_cast<char>(ch));
                }
                break;
        }
    }
}

void ESQLShell::run_termux() {
    if (use_colors) {
        std::cout << BLUE << "Termux mode activated (using line-based input)\n\n" << RESET;
    }
    
    while (true) {
        print_prompt();
        
        std::string line;
        std::getline(std::cin, line);
        
        if (line.empty()) continue;
        if (line == "exit" || line == "quit") {
            disable_raw_mode();
            exit(0);
        }
        
        execute_command(line);
    }
}

void ESQLShell::redraw_interface() {
    if (is_termux()) return;
    
    // Clear current line and redraw
    std::cout << "\r\033[K";
    print_prompt();
    std::cout << colorize_sql(current_line);
    
    // Position cursor correctly
    int prompt_length = get_current_time().length() + current_db.length() + 8;
    std::cout << "\033[" << (prompt_length + cursor_pos) << "G";
    std::cout.flush();
}

void ESQLShell::print_banner() {
    clear_screen();
    if (use_colors) {
        std::cout << GRAY;
        std::cout << "   ███████╗███████╗ ██████╗ ██╗   \n";
        std::cout << "   ██╔════╝██╔════╝██╔═══██╗██║   \n";
        std::cout << "   █████╗  ███████╗██║   ██║██║   \n";
        std::cout << "   ██╔══╝  ╚════██║██║   ██║██║   \n";
        std::cout << "   ███████╗███████║╚██████╔╝███████╗\n";
        std::cout << "   ╚══════╝╚══════╝ ╚═════╝ ╚══════╝\n";
        std::cout << RESET;
        
        std::cout << CYAN << "╔═══════════════════════════════════════╗\n";
        std::cout << "║    " << MAGENTA << "E N H A N C E D   ES Q L   S H E L L" << CYAN << "  ║\n";
        std::cout << "║        " << YELLOW << "H4CK3R  STYL3  V3RSI0N" << CYAN << "         ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n" << RESET;

        std::cout << RED << "[*] "<< CYAN <<  "Type 'help' for commands, 'exit' to quit\n";
        std::cout << RED << "[*] " << CYAN << "Initializing ESQL Database Matrix...\n";
        std::cout << RED << "[*] " << MAGENTA << "Quantum ESQL Processor: ONLINE\n";
        std::cout << RED << "[*] " << GRAY << "Syntax Highlighting: ACTIVATED\n"<<RESET;
        std::cout << MAGENTA << "[+] "<< CYAN << "Connected to: " << (use_colors ? GRAY : "") 
                  << current_db <<GREEN << "•" << (use_colors ? RESET : "") << "\n\n";
    }
}

void ESQLShell::print_prompt() {
    if (is_termux()) {
        std::cout << YELLOW << "[" << get_current_time() << "] "
                  << GREEN << "• " 
                  << GRAY << current_db 
                  << RESET << "> ";
    } else {
        std::cout << (use_colors ? YELLOW : "") << "[" << get_current_time() << "] "
                  << (use_colors ? GREEN : "") << "• " 
                  << (use_colors ? GRAY : "") << current_db 
                  << (use_colors ? RESET : "") << "> ";
    }
}

std::string ESQLShell::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M");
    return ss.str();
}

std::string ESQLShell::colorize_sql(const std::string& input) {
    std::string result;
    bool in_string = false;
    char string_delim = 0;
    std::string current_word;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (in_string) {
            result += GREEN + std::string(1, c);
            if (c == string_delim) {
                in_string = false;
                result += RESET;
            }
            continue;
        }

        if (c == '"' || c == '\'') {
            if (!current_word.empty()) {
                std::string upper_word = current_word;
                std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);

                if (keywords.find(upper_word) != keywords.end()) {
                    result += MAGENTA + current_word + RESET;
                } else if (datatypes.find(upper_word) != datatypes.end()) {
                    result += BLUE + current_word + RESET;
                } else if (conditionals.find(upper_word) != conditionals.end()) {
                    result += CYAN + current_word + RESET;
                } else {
                    result += current_word;
                }
                current_word.clear();
            }

            in_string = true;
            string_delim = c;
            result += GREEN + std::string(1, c);
            continue;
        }

        if (std::isspace(c) || c == ',' || c == ';' || c == '(' || c == ')' ||
            c == '=' || c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/') {
            if (!current_word.empty()) {
                std::string upper_word = current_word;
                std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);

                if (keywords.find(upper_word) != keywords.end()) {
                    result += MAGENTA + current_word + RESET;
                } else if (datatypes.find(upper_word) != datatypes.end()) {
                    result += BLUE + current_word + RESET;
                } else if (conditionals.find(upper_word) != conditionals.end()) {
                    result += CYAN + current_word + RESET;
                } else {
                    result += current_word;
                }
                current_word.clear();
            }
            result += std::string(1, c);
            continue;
        }

        current_word += c;
    }

    if (!current_word.empty()) {
        std::string upper_word = current_word;
        std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);

        if (keywords.find(upper_word) != keywords.end()) {
            result += MAGENTA + current_word + RESET;
        } else if (datatypes.find(upper_word) != datatypes.end()) {
            result += BLUE + current_word + RESET;
        } else if (conditionals.find(upper_word) != conditionals.end()) {
            result += CYAN + current_word + RESET;
        } else {
            result += current_word;
        }
    }

    return result;
}

bool ESQLShell::is_termux() const {
    return current_platform == Platform::Termux;
}

void ESQLShell::insert_char(char c) {
    current_line.insert(cursor_pos, 1, c);
    cursor_pos++;
    redraw_interface();
}

void ESQLShell::delete_char() {
    if (cursor_pos > 0 && !current_line.empty()) {
        current_line.erase(cursor_pos - 1, 1);
        cursor_pos--;
        redraw_interface();
    }
}

void ESQLShell::move_cursor_left() {
    if (cursor_pos > 0) {
        cursor_pos--;
        redraw_interface();
    }
}

void ESQLShell::move_cursor_right() {
    if (cursor_pos < current_line.length()) {
        cursor_pos++;
        redraw_interface();
    }
}

void ESQLShell::navigate_history_up() {
    if (command_history.empty()) return;
    
    if (history_index == -1) {
        history_index = command_history.size() - 1;
    } else if (history_index > 0) {
        history_index--;
    }
    
    current_line = command_history[history_index];
    cursor_pos = current_line.length();
    redraw_interface();
}

void ESQLShell::navigate_history_down() {
    if (command_history.empty()) return;
    
    if (history_index < static_cast<int>(command_history.size()) - 1) {
        history_index++;
        current_line = command_history[history_index];
    } else {
        history_index = command_history.size();
        current_line.clear();
    }
    cursor_pos = current_line.length();
    redraw_interface();
}

void ESQLShell::handle_tab_completion() {
    auto suggestions = get_completion_suggestion(current_line);
    if (suggestions.empty()) return;

    if (suggestions.size() == 1) {
        current_line = suggestions[0];
        cursor_pos = current_line.length();
    } else {
        std::cout << "\n";
        for (const auto& suggestion : suggestions) {
            std::cout << " " << colorize_sql(suggestion) << "\n";
        }
        std::cout << "\n";
        redraw_interface();
    }
}

std::vector<std::string> ESQLShell::get_completion_suggestion(const std::string& input) {
    std::vector<std::string> suggestions;
    std::string last_word;
    
    size_t last_space = input.find_last_of(" \t\n,;()");
    if (last_space == std::string::npos) {
        last_word = input;
    } else {
        last_word = input.substr(last_space + 1);
    }
    
    std::string upper_last_word = last_word;
    std::transform(upper_last_word.begin(), upper_last_word.end(), upper_last_word.begin(), ::toupper);
    
    for (const auto& kw : keywords) {
        if (kw.find(upper_last_word) == 0) {
            std::string completion = input.substr(0, last_space + 1) + kw;
            suggestions.push_back(completion);
        }
    }
    
    for (const auto& dt : datatypes) {
        if (dt.find(upper_last_word) == 0) {
            std::string completion = input.substr(0, last_space + 1) + dt;
            suggestions.push_back(completion);
        }
    }
    
    for (const auto& fn : conditionals) {
        if (fn.find(upper_last_word) == 0) {
            std::string completion = input.substr(0, last_space + 1) + fn;
            suggestions.push_back(completion);
        }
    }
    
    return suggestions;
}

void ESQLShell::handle_enter() {
    std::cout << "\n";
    
    if (!current_line.empty()) {
        add_to_history(current_line);
        execute_command(current_line);
    }
    
    current_line.clear();
    cursor_pos = 0;
    history_index = -1;
}

void ESQLShell::add_to_history(const std::string& command) {
    if (!command.empty() && (command_history.empty() || command_history.back() != command)) {
        command_history.push_back(command);
    }
}

void ESQLShell::execute_command(const std::string& command) {
    if (command.empty()) return;
    
    add_to_history(command);
    
    std::string upper_cmd = command;
    std::transform(upper_cmd.begin(), upper_cmd.end(), upper_cmd.begin(), ::toupper);
    
    if (upper_cmd == "EXIT" || upper_cmd == "QUIT") {
        db.shutdown();
        disable_raw_mode();
        exit(0);
    } else if (upper_cmd == "HELP") {
        show_help();
    } else if (upper_cmd == "CLEAR") {
        clear_screen();
        print_banner();
    } else {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            auto [result, duration] = db.executeQuery(command);
            auto end = std::chrono::high_resolution_clock::now();
            double actual_duration = std::chrono::duration<double>(end - start).count();
            
            print_results(result, actual_duration);
        } catch (const std::exception& e) {
            std::cerr << (use_colors ? RED : "") << "Error: " << e.what() 
                      << (use_colors ? RESET : "") << "\n";
        }
    }
    
    std::cout << "\n";
}

void ESQLShell::print_results(const ExecutionEngine::ResultSet& result, double duration) {
    if (result.columns.empty()) {
        std::cout << (use_colors ? GREEN : "") << "Query executed successfully.\n"
                  << (use_colors ? RESET : "");
        return;
    }
    
    // Calculate column widths
    std::vector<size_t> widths(result.columns.size());
    for (size_t i = 0; i < result.columns.size(); ++i) {
        widths[i] = result.columns[i].length() + 2;
        for (const auto& row : result.rows) {
            if (i < row.size() && row[i].length() + 2 > widths[i]) {
                widths[i] = row[i].length() + 2;
            }
        }
    }
    
    // Print header
    std::cout << (use_colors ? CYAN : "");
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::left << std::setw(widths[i]) << result.columns[i] << " |";
    }
    std::cout << (use_colors ? RESET : "") << "\n";
    
    // Print separator
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-') << "-|";
    }
    std::cout << "\n";
    
    // Print rows
    for (const auto& row : result.rows) {
        std::cout << (use_colors ? MAGENTA : "");
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << std::left << std::setw(widths[i]) << row[i] << " |";
        }
        std::cout << (use_colors ? RESET : "") << "\n";
    }
    
    // Print summary
    std::cout << "(" << result.rows.size() << " row" 
              << (result.rows.size() != 1 ? "s" : "") << ")\n";
    std::cout << (use_colors ? GRAY : "") << "Time: " << std::fixed 
              << std::setprecision(4) << duration << " seconds\n"
              << (use_colors ? RESET : "");
}

void ESQLShell::show_help() {
    std::cout << "\n" << (use_colors ? CYAN : "") << "Available commands:" 
              << (use_colors ? RESET : "") << "\n";
    
    const std::vector<std::pair<std::string, std::string>> commands = {
        {"SELECT", "Query data from tables"},
        {"INSERT", "Add new records"},
        {"UPDATE", "Modify existing records"},
        {"DELETE", "Remove records"},
        {"CREATE", "Create tables/databases"},
        {"DROP", "Remove tables/databases"},
        {"ALTER", "Modify table structure"},
        {"USE", "Switch databases"},
        {"SHOW", "Display database info"},
        {"DESCRIBE", "Show table structure"},
        {"HELP", "Show this help"},
        {"EXIT/QUIT", "Quit the shell"},
        {"CLEAR", "Clear the screen"}
    };
    
    for (const auto& [cmd, desc] : commands) {
        std::cout << "  " << (use_colors ? MAGENTA : "") << cmd 
                  << (use_colors ? GREEN : "") << " - " << desc 
                  << (use_colors ? RESET : "") << "\n";
    }
    std::cout << "\n";
}

void ESQLShell::setCurrentDatabase(const std::string& db_name) {
    current_db = db_name;
}
