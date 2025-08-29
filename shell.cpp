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
    "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP"
};

const std::unordered_set<std::string> ESQLShell::datatypes = {
    "INT", "INTEGER", "FLOAT", "TEXT", "STRING", "BOOL", "BOLEAN", "VARCHAR", "DATE"
};

const std::unordered_set<std::string> ESQLShell::tables = {
    "users", "orders", "products", "customers"
};

const std::unordered_set<std::string> ESQLShell::conditionals = {
    "AND", "OR", "NOT", "NULL", "IS", "LIKE", "IN", "BETWEEN"
};

ESQLShell::ESQLShell(Database& db) : db(db), current_db("default") {
    platform_specific_init();
    enable_raw_mode();
    get_terminal_size();
    initialize_terminal();
}

ESQLShell::~ESQLShell() {
    disable_raw_mode();
    std::cout << RESET << std::endl;
}

ESQLShell::Platform ESQLShell::detect_platform() {
    #ifdef _WIN32
    return Platform::Windows;
    #else
    // Simple and reliable Termux detection
    const char* term = getenv("TERM");
    const char* pkg = getenv("PREFIX");
    
    // Check if we're running in Termux
    if (pkg && std::string(pkg).find("com.termux") != std::string::npos) {
        return Platform::Termux;
    }
    
    // Check common Termux files
    if (access("/data/data/com.termux/files/usr/bin/login", F_OK) == 0) {
        return Platform::Termux;
    }
    
    if (access("/data/data/com.termux/files/home", F_OK) == 0) {
        return Platform::Termux;
    }
    
    // Check ANDROID_ROOT which is set on Android
    const char* android_root = getenv("ANDROID_ROOT");
    if (android_root) {
        return Platform::Termux;
    }
    
    return Platform::Linux;
    #endif
}

void ESQLShell::platform_specific_init() {
    current_platform = detect_platform();

    #ifdef _WIN32
    hInput = GetStdHandle(STD_INPUT_HANDLE);
    hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleMode(hOutput, &original_mode);
    SetConsoleMode(hOutput, original_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    #else
    if (current_platform != Platform::Termux) {
        // Only set raw mode for non-Termux platforms
        struct termios term;
        tcgetattr(STDIN_FILENO, &term);
        term.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &term);
    }
    #endif

    // Enable colors for all platforms including Termux
    use_colors = true;
}

void ESQLShell::enable_raw_mode() {
    #ifdef _WIN32
    DWORD mode = 0;
    GetConsoleMode(hInput, &mode);
    SetConsoleMode(hInput, mode & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
    #else
    if (!is_termux()) {
        tcgetattr(STDIN_FILENO, &orig_termios);
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    }
    #endif
}

void ESQLShell::disable_raw_mode() {
    #ifdef _WIN32
    SetConsoleMode(hInput, original_mode);
    SetConsoleMode(hOutput, original_mode);
    #else
    if (!is_termux()) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    }
    #endif
}

void ESQLShell::get_terminal_size() {
    #ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(hOutput, &csbi);
    terminal_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    terminal_height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    #else
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    if (ws.ws_col > 0) terminal_width = ws.ws_col;
    if (ws.ws_row > 0) terminal_height = ws.ws_row;
    #endif
}

void ESQLShell::clear_screen() {
    std::cout << "\033[2J\033[H";
}

void ESQLShell::clear_previous_lines(int line_count) {
    if (line_count <= 0) return;
    
    std::cout << "\033[" << line_count << "A\033[2K";
    for (int i = 1; i < line_count; ++i) {
        std::cout << "\033[1B\033[2K";
    }
    std::cout << "\033[0G";
}

int ESQLShell::read_key() {
    if (is_termux()) {
        // Termux uses simple input
        return -1;
    }
    
    #ifdef _WIN32
    INPUT_RECORD ir;
    DWORD count;
    
    while (true) {
        ReadConsoleInput(hInput, &ir, 1, &count);
        if (ir.EventType == KEY_EVENT && ir.Event.KeyEvent.bKeyDown) {
            char c = ir.Event.KeyEvent.uChar.AsciiChar;
            if (c != 0) return c;
            
            switch (ir.Event.KeyEvent.wVirtualKeyCode) {
                case VK_LEFT: return 1000;
                case VK_RIGHT: return 1001;
                case VK_UP: return 1002;
                case VK_DOWN: return 1003;
                case VK_PRIOR: return 1004;
                case VK_NEXT: return 1005;
                case VK_TAB: return '\t';
                case VK_BACK: return 8;
                case VK_RETURN: return '\n';
                case VK_ESCAPE: return 27;
            }
        }
    }
    #else
    char c;
    if (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == 27) {
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1) {
                if (seq[0] == '[') {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1) {
                        switch (seq[1]) {
                            case 'D': return 1000;
                            case 'C': return 1001;
                            case 'A': return 1002;
                            case 'B': return 1003;
                        }
                    }
                }
            }
            return 27;
        }
        return c;
    }
    return -1;
    #endif
}

void ESQLShell::run() {
    initialize_terminal();
    print_banner();
    
    // Termux requires special handling
    if (is_termux()) {
        run_termux();
    } else {
        run_standard();
    }
}

// Standard terminal mode (Linux/Windows)
void ESQLShell::run_standard() {
    while (true) {
        print_prompt();
        redraw_interface();
        
        int ch = read_key();
        
        switch (ch) {
            case 27: // ESC
                if (multi_line_mode) {
                    multi_line_mode = false;
                    current_line.clear();
                    input_lines.clear();
                    current_line_index = 0;
                } else {
                    return;
                }
                break;
                
            case '\n': case '\r':
                handle_enter();
                break;
                
            case 8: case 127: // Backspace
                delete_char();
                break;
                
            case '\t': // Tab
                handle_tab_completion();
                break;
                
            case 1000: // Left
                move_cursor_left();
                break;
                
            case 1001: // Right
                move_cursor_right();
                break;
                
            case 1002: // Up
                if (multi_line_mode && current_line_index > 0) {
                    current_line_index--;
                    current_line = input_lines[current_line_index];
                    cursor_pos = current_line.length();
                } else {
                    navigate_history_up();
                }
                break;
                
            case 1003: // Down
                if (multi_line_mode && current_line_index < input_lines.size() - 1) {
                    current_line_index++;
                    current_line = input_lines[current_line_index];
                    cursor_pos = current_line.length();
                } else {
                    navigate_history_down();
                }
                break;
                
            default:
                if (ch >= 32 && ch <= 126) {
                    insert_char(static_cast<char>(ch));
                }
                break;
        }
    }
}

// Termux-specific implementation - uses line-based input like bash
void ESQLShell::run_termux() {
    if (use_colors) {
        std::cout << BLUE << "Termux mode activated (using line-based input with ANSI colors)\n\n" << RESET;
    } else {
        std::cout << "Termux mode activated (using line-based input)\n\n";
    }
    
    while (true) {
        print_prompt();
        
        // For Termux, we use simple line-based input
        std::string line;
        std::getline(std::cin, line);
        
        if (line.empty()) continue;
        if(line=="exit" || line=="quit"){
		disable_raw_mode();
		exit(0);
	}
        // Handle multi-line input
        if (multi_line_mode) {
            input_lines.push_back(line);
            
            // Check if we should end multi-line mode
            if (line.find(';') != std::string::npos) {
                std::string full_command;
                for (const auto& input_line : input_lines) {
                    full_command += input_line + " ";
                }
                
                execute_command(full_command);
                multi_line_mode = false;
                input_lines.clear();
            } else {
                std::cout << "> "; // Continuation prompt
            }
        } else {
            // Check if this should be multi-line
            if (line.find(';') == std::string::npos && 
                !is_single_line_command(line)) {
                multi_line_mode = true;
                input_lines.push_back(line);
                std::cout << "> "; // Continuation prompt
            } else {
                execute_command(line);
            }
        }
    }
}

void ESQLShell::redraw_interface() {
    if (is_termux()) {
        // Termux doesn't need complex redrawing
        return;
    }
    
    int line_count = get_line_count();
    
    // Clear previous lines
    if (line_count > 0) {
        std::cout << "\033[" << line_count << "A"; // Move up
        for (int i = 0; i < line_count; ++i) {
            std::cout << "\033[2K"; // Clear line
            if (i < line_count - 1) {
                std::cout << "\033[1B"; // Move down for next line
            }
        }
        std::cout << "\033[" << line_count << "A"; // Move back to start
    }
    
    // Print prompt and colored input
    print_prompt();
    if (!is_termux()) {
        std::cout << colorize_sql();
    } else {
        std::cout << current_line;
    }

    // Calculate cursor position
    int cursor_line, cursor_col;
    get_cursor_position(cursor_line, cursor_col);

    // Move cursor to correct position
    std::cout << "\033[" << (cursor_line + 1) << ";" << (cursor_col + 1) << "H";
    std::cout.flush();
}

void ESQLShell::initialize_terminal() {
    if (is_termux()) {
        // Minimal initialization for Termux with ANSI colors
        std::cout << "\033[0m"; // Reset colors
        std::cout << "\033[?25h"; // Show cursor
        return;
    }
    
    // Standard initialization for other platforms
    std::cout << "\033c";
    std::cout << "\033[?25h";
    std::cout << "\033[0m";
    
    #ifndef _WIN32
    system("stty sane 2>/dev/null");
    #endif
}

void ESQLShell::print_banner() {
    clear_screen();
    if (use_colors) {
        std::cout << BLUE << "\n";
        std::cout << "   _____ ______ ____  _      \n";
        std::cout << "  | ____|  ____/ ___|| |     \n";
        std::cout << "  |  _| | |__  \\___ \\| |     \n";
        std::cout << "  | |___|  __|  ___) | |___  \n";
        std::cout << "  |_____|_|    |____/|_____| \n";
        std::cout << RESET << "  Enhanced SQL Shell v2.0\n\n";
    } else {
        std::cout << "\nESQL Enhanced SQL Shell v2.0\n\n";
    }
    
    std::cout << "Type 'help' for commands, 'exit' to quit\n";
    std::cout << "Connected to: " << (use_colors ? GREEN : "") << current_db 
              << (use_colors ? RESET : "") << "\n\n";
}

void ESQLShell::print_prompt() {
    if (is_termux()) {
        // Colorful prompt for Termux with ANSI colors
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

std::string ESQLShell::colorize_sql() {
    std::string result;
    bool in_string = false;
    char string_delim = 0;
    std::string current_word;
    
    for (size_t i = 0; i < current_line.size(); ++i) {
        char c = current_line[i];
        
        if (in_string) {
            result += (use_colors ? GRAY : "") + std::string(1, c);
            if (c == string_delim) {
                in_string = false;
                if (use_colors) result += RESET;
            }
            continue;
        }
        
        if (c == '"' || c == '\'') {
            if (!current_word.empty()) {
                // Process the current word before the quote
                std::string upper_word = current_word;
                std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
                
                if (keywords.find(upper_word) != keywords.end()) {
                    result += (use_colors ? MAGENTA : "") + current_word + (use_colors ? RESET : "");
                } else if (datatypes.find(upper_word) != datatypes.end()) {
                    result += (use_colors ? BLUE : "") + current_word + (use_colors ? RESET : "");
                } else if (conditionals.find(upper_word) != conditionals.end()) {
                    result += (use_colors ? CYAN : "") + current_word + (use_colors ? RESET : "");
                } else if (i >= cursor_pos && !is_valid_token(current_word)) {
                    result += (use_colors ? RED : "") + current_word + (use_colors ? RESET : "");
                } else {
                    result += current_word;
                }
                current_word.clear();
            }
            
            in_string = true;
            string_delim = c;
            result += (use_colors ? GRAY : "") + std::string(1, c);
            continue;
        }
        
        if (std::isspace(c) || c == ',' || c == ';' || c == '(' || c == ')' || 
            c == '=' || c == '<' || c == '>') {
            if (!current_word.empty()) {
                std::string upper_word = current_word;
                std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
                
                if (keywords.find(upper_word) != keywords.end()) {
                    result += (use_colors ? MAGENTA : "") + current_word + (use_colors ? RESET : "");
                } else if (datatypes.find(upper_word) != datatypes.end()) {
                    result += (use_colors ? BLUE : "") + current_word + (use_colors ? RESET : "");
                } else if (conditionals.find(upper_word) != conditionals.end()) {
                    result += (use_colors ? CYAN : "") + current_word + (use_colors ? RESET : "");
                } else if (i >= cursor_pos && !is_valid_token(current_word)) {
                    result += (use_colors ? RED : "") + current_word + (use_colors ? RESET : "");
                } else {
                    result += current_word;
                }
                current_word.clear();
            }
            result += (use_colors ? RESET : "") + std::string(1, c);
            continue;
        }
        
        current_word += c;
    }
    
    if (!current_word.empty()) {
        std::string upper_word = current_word;
        std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
        
        if (keywords.find(upper_word) != keywords.end()) {
            result += (use_colors ? MAGENTA : "") + current_word + (use_colors ? RESET : "");
        } else if (datatypes.find(upper_word) != datatypes.end()) {
            result += (use_colors ? BLUE : "") + current_word + (use_colors ? RESET : "");
        } else if (conditionals.find(upper_word) != conditionals.end()) {
            result += (use_colors ? CYAN : "") + current_word + (use_colors ? RESET : "");
        } else if (current_line.size() >= cursor_pos && !is_valid_token(current_word)) {
            result += (use_colors ? RED : "") + current_word + (use_colors ? RESET : "");
        } else {
            result += current_word;
        }
    }
    
    // Ensure we reset colors at the end
    if (use_colors) {
        result += RESET;
    }
    
    return result;
}

bool ESQLShell::is_valid_token(const std::string& token) const {
    if (token.empty()) return true;
    if (std::isdigit(token[0])) return false;
    return true;
}

bool ESQLShell::is_termux() const {
    return current_platform == Platform::Termux;
}

void ESQLShell::insert_char(char c) {
    current_line.insert(cursor_pos, 1, c);
    cursor_pos++;
}

void ESQLShell::delete_char() {
    if (cursor_pos > 0 && !current_line.empty()) {
        current_line.erase(cursor_pos - 1, 1);
        cursor_pos--;
    }
}

void ESQLShell::move_cursor_left() {
    if (cursor_pos > 0) cursor_pos--;
}

void ESQLShell::move_cursor_right() {
    if (cursor_pos < current_line.length()) cursor_pos++;
}

void ESQLShell::move_cursor_up() {
    // Simplified up movement for multiline
    if (multi_line_mode && current_line_index > 0) {
        current_line_index--;
        current_line = input_lines[current_line_index];
        cursor_pos = current_line.length();
    }
}

void ESQLShell::move_cursor_down() {
    // Simplified down movement for multiline
    if (multi_line_mode && current_line_index < input_lines.size() - 1) {
        current_line_index++;
        current_line = input_lines[current_line_index];
        cursor_pos = current_line.length();
    }
}

void ESQLShell::navigate_history_up() {
    if (history_index > 0) {
        history_index--;
        current_line = command_history[history_index];
        cursor_pos = current_line.length();
    }
}

void ESQLShell::navigate_history_down() {
    if (history_index < static_cast<int>(command_history.size()) - 1) {
        history_index++;
        current_line = command_history[history_index];
        cursor_pos = current_line.length();
    } else if (history_index == static_cast<int>(command_history.size()) - 1) {
        history_index++;
        current_line.clear();
        cursor_pos = 0;
    }
}

void ESQLShell::handle_tab_completion() {
    std::string completed = complete(current_line);
    if (completed != current_line) {
        current_line = completed;
        cursor_pos = current_line.length();
    }
}

void ESQLShell::handle_enter() {
    if (multi_line_mode) {
        // Add current line to multiline input
        input_lines.push_back(current_line);
        current_line_index++;
        
        // Check if we should end multiline mode (ends with semicolon)
        if (current_line.find(';') != std::string::npos) {
            // Combine all lines and execute
            std::string full_command;
            for (const auto& line : input_lines) {
                full_command += line + " ";
            }
            
            execute_command(full_command);
            
            // Reset multiline state
            multi_line_mode = false;
            input_lines.clear();
            current_line_index = 0;
            current_line.clear();
            cursor_pos = 0;
        } else {
            // Continue multiline input
            std::cout << "\n";
            current_line.clear();
            cursor_pos = 0;
        }
    } else {
        // Single line mode
        if (current_line.find(';') == std::string::npos && 
            !current_line.empty() && 
            !is_single_line_command(current_line)) {
            // Start multiline mode
            multi_line_mode = true;
            input_lines.push_back(current_line);
            current_line_index = 1;
            std::cout << "\n";
            current_line.clear();
            cursor_pos = 0;
        } else {
            // Execute single line command
            execute_command(current_line);
            current_line.clear();
            cursor_pos = 0;
        }
    }
}

bool ESQLShell::is_single_line_command(const std::string& command) const {
    std::string upper_cmd = command;
    std::transform(upper_cmd.begin(), upper_cmd.end(), upper_cmd.begin(), ::toupper);
    
    return upper_cmd == "EXIT" || upper_cmd == "QUIT" || 
           upper_cmd == "HELP" || upper_cmd == "CLEAR" ||
           upper_cmd.find("USE ") == 0;
}

// In ESQLShell::execute_command() method in shell.cpp
void ESQLShell::execute_command(const std::string& command) {
    std::cout << "\n";
    
    if (!command.empty()) {
        add_to_history(command);
        
        std::string upper_cmd = command;
        std::transform(upper_cmd.begin(), upper_cmd.end(), upper_cmd.begin(), ::toupper);
        
        if (upper_cmd == "EXIT" || upper_cmd == "QUIT") {
            disable_raw_mode();
            exit(0);
        } else if (upper_cmd == "HELP") {
            show_help();
        } else if (upper_cmd == "CLEAR") {
            clear_screen();
            print_banner();
        } else {
            try {
                // Handle USE command specifically to avoid double processing
                if (upper_cmd.find("USE ") == 0) {
                    size_t pos = command.find(' ');
                    if (pos != std::string::npos) {
                        std::string db_name = command.substr(pos + 1);
                        // Remove trailing semicolon if present
                        if (!db_name.empty() && db_name.back() == ';') {
                            db_name.pop_back();
                        }
                        // Trim whitespace and remove "DATABASE" keyword if present
                        db_name.erase(0, db_name.find_first_not_of(" \t\n"));
                        db_name.erase(db_name.find_last_not_of(" \t\n") + 1);
                        
                        // Remove "DATABASE" keyword if it exists
                        std::string upper_db_name = db_name;
                        std::transform(upper_db_name.begin(), upper_db_name.end(), upper_db_name.begin(), ::toupper);
                        if (upper_db_name.find("DATABASE ") == 0) {
                            db_name = db_name.substr(9); // Remove "DATABASE "
                            db_name.erase(0, db_name.find_first_not_of(" \t\n"));
                        }
                        
                        if (!db_name.empty()) {
                            // Use the database through proper channels - FIXED HERE
                            db.setCurrentDatabase(db_name);
                            // ADD THIS LINE to update the storage layer:
                            db.storage->useDatabase(db_name);
                            current_db = db_name;
                            std::cout << "Switched to database: " << db_name << "\n";
                        }
                    }
                } else {
                    // Execute all other commands normally
                    auto start = std::chrono::high_resolution_clock::now();
                    auto [result, duration] = db.executeQuery(command);
                    auto end = std::chrono::high_resolution_clock::now();
                    double actual_duration = std::chrono::duration<double>(end - start).count();
                    
                    print_results(result, actual_duration);
                }
                
            } catch (const std::exception& e) {
                std::cerr << (use_colors ? RED : "") << "Error: " << e.what() 
                          << (use_colors ? RESET : "") << "\n";
            }
        }
        
        std::cout << "\n";
    }
}

void ESQLShell::add_to_history(const std::string& command) {
    if (!command.empty() && (command_history.empty() || command_history.back() != command)) {
        command_history.push_back(command);
    }
    history_index = command_history.size();
}

std::string ESQLShell::complete(const std::string& input) {
    size_t pos = input.find_last_of(" ,;\n");
    std::string last_word = (pos == std::string::npos) ? input : input.substr(pos + 1);
    
    for (const auto& kw : keywords) {
        if (kw.find(last_word) == 0) {
            return input.substr(0, pos + 1) + kw;
        }
    }
    
    for (const auto& dt : datatypes) {
        if (dt.find(last_word) == 0) {
            return input.substr(0, pos + 1) + dt;
        }
    }
    
    for (const auto& tbl : tables) {
        if (tbl.find(last_word) == 0) {
            return input.substr(0, pos + 1) + tbl;
        }
    }
    
    return input;
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
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << std::left << std::setw(widths[i]) << row[i] << " |";
        }
        std::cout << "\n";
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
                  << (use_colors ? WHITE : "") << " - " << desc 
                  << (use_colors ? RESET : "") << "\n";
    }
    std::cout << "\n";
}

void ESQLShell::get_cursor_position(int& line, int& col) const {
    line = 0;
    col = 0;
    
    // Calculate prompt length (time + db name + fixed characters)
    int prompt_length = get_current_time().length() + current_db.length() + 6;
    int current_col = prompt_length;
    
    for (size_t i = 0; i < cursor_pos; ++i) {
        if (current_line[i] == '\n') {
            line++;
            current_col = 0;
        } else {
            if (current_col >= terminal_width - 1) {
                line++;
                current_col = 1;
            }
            current_col++;
        }
    }
    
    col = current_col;
}

int ESQLShell::get_line_count() const {
    int lines = 1;
    int prompt_length = get_current_time().length() + current_db.length() + 6;
    int current_col = prompt_length;

    for (char c : current_line) {
        if (c == '\n') {
            lines++;
            current_col = 0;
        } else {
            if (current_col >= terminal_width - 1) {
                lines++;
                current_col = 1;
            }
            current_col++;
        }
    }

    return lines;
}

void ESQLShell::setCurrentDatabase(const std::string& db_name) {
    current_db = db_name;
}
