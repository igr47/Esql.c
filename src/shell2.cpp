// shell.cpp
#include "shell2.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <thread>
#include <cstring>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#endif

// Static member initialization
volatile sig_atomic_t ESQLShell::terminal_resized = 0;

// Signal handler for terminal resize
void handle_winch(int sig) {
    ESQLShell::terminal_resized = 1;
}

// SQL elements for syntax highlighting
const std::unordered_set<std::string> ESQLShell::keywords = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DATABASE", "DROP", "ALTER", "ADD", "RENAME",
    "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP", "DISTINCT", "DATABASES","BY","ORDER","GROUP","HAVING","BULK","ROW","TABLES","STRUCTURE"
};

const std::unordered_set<std::string> ESQLShell::datatypes = {
    "INT", "INTEGER", "FLOAT", "TEXT", "STRING", "BOOL", "BOLEAN", "VARCHAR", "DATE", "DATETIME", "UUID"
};

const std::unordered_set<std::string> ESQLShell::conditionals = {
    "AND", "OR", "NOT", "NULL", "IS", "LIKE", "IN", "BETWEEN", "OFFSET", "LIMIT","AS","PRIMARY_KEY","UNIQUE","DEFAULT","AUTO_INCREAMENT","CHECK","NOT_NULL","TO","TRUE","FALSE", "GENERATE_UUID", "GENERATE_DATE", "GENERATE_DATE_TIME", "CASE", "WHEN", "THEN", "ELSE"
};

ESQLShell::ESQLShell(Database& db) : db(db), current_db("default") {
    current_platform = detect_platform();
    get_terminal_size();
    
    if (!is_termux()) {
        enable_raw_mode();
        setup_signal_handlers();
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

void ESQLShell::setup_signal_handlers() {
    #ifndef _WIN32
    struct sigaction sa;
    sa.sa_handler = handle_winch;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGWINCH, &sa, nullptr);
    #endif
}

void ESQLShell::handle_terminal_resize() {
    if (terminal_resized) {
        get_terminal_size();
        redraw_interface();
        terminal_resized = 0;
    }
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
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        terminal_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    } else {
        terminal_width = 80; // fallback
    }
    #else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        terminal_width = ws.ws_col;
    } else {
        // Try environment variables as fallback
        const char* columns = getenv("COLUMNS");
        if (columns) {
            terminal_width = std::atoi(columns);
        } else {
            terminal_width = 80; // default fallback
        }
    }
    #endif

    // Ensure minimum width
    if (terminal_width < 40) terminal_width = 40;
}

size_t ESQLShell::visible_length(const std::string& str) const {
    size_t len = 0;
    bool in_escape = false;
    
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == '\033') {
            in_escape = true;
        } else if (in_escape) {
            if (str[i] == 'm') {
                in_escape = false;
            }
        } else {
            len++;
        }
    }
    return len;
}

size_t ESQLShell::find_word_boundary(const std::string& text, size_t max_visible, size_t start_pos) const {
    size_t visible_count = 0;
    size_t last_whitespace = start_pos;
    bool in_escape = false;
    
    for (size_t i = start_pos; i < text.length(); i++) {
        char c = text[i];
        
        // Handle escape sequences
        if (c == '\033') {
            in_escape = true;
            continue;
        } else if (in_escape) {
            if (c == 'm') in_escape = false;
            continue;
        }
        
        // Count visible characters
        visible_count++;
        
        // Track last whitespace for word breaking
        if (std::isspace(c)) {
            last_whitespace = i + 1; // Position after whitespace
        }
        
        // If we've reached the max width
        if (visible_count >= max_visible) {
            // Prefer breaking at word boundary
            if (last_whitespace > start_pos && (last_whitespace - start_pos) > max_visible * 0.3) {
                return last_whitespace;
            }
            // Otherwise break at current position
            return i;
        }
    }
    
    return text.length();
}

std::vector<std::string> ESQLShell::wrap_line_with_colors(const std::string& input, int first_line_width) const {
    std::vector<std::string> lines;
    std::string remaining = input;
    int current_width = first_line_width;
    
    while (!remaining.empty()) {
        int max_width = terminal_width - current_width;
        if (max_width < 10) max_width = 10; // Minimum width for continuation
        
        size_t break_pos = find_word_boundary(remaining, max_width);
        
        // If we're at the end of string, take everything
        if (break_pos >= remaining.length()) {
            lines.push_back(remaining);
            break;
        }
        
        std::string line = remaining.substr(0, break_pos);
        lines.push_back(line);
        
        // Remove the broken part
        remaining = remaining.substr(break_pos);
        
        // Trim leading whitespace from continuation lines
        if (!remaining.empty()) {
            size_t first_non_ws = remaining.find_first_not_of(" \t");
            if (first_non_ws != std::string::npos && first_non_ws > 0) {
                remaining = remaining.substr(first_non_ws);
            }
        }
        
        // Continuation lines have 2-space indent
        current_width = 2;
    }
    
    return lines;
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
        if (ch == -1) {
            // Check for terminal resize even when no key is pressed
            handle_terminal_resize();
            continue;
        }
        
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

    handle_terminal_resize();

    std::string prompt_str = "[" + get_current_time() + "] • " + current_db + "> ";
    int prompt_length = visible_length(prompt_str);

    // Clear from cursor to end of screen to remove any old wrapped lines
    std::cout << "\r\033[K"; // Clear current line

    // Print prompt
    print_prompt();

    if (current_line.empty()) {
        std::cout.flush();
        return;
    }

    // Colorize the input
    std::string colored_line = colorize_sql(current_line);

    // Calculate available width for first line (after prompt)
    int available_width = terminal_width - prompt_length;
    if (available_width < 10) available_width = 10;

    // Break the colored line into properly wrapped lines
    std::vector<std::string> display_lines;
    std::string remaining = colored_line;
    int current_line_width = available_width; // First line has prompt space

    while (!remaining.empty()) {
        if (remaining.length() <= current_line_width) {
            // Last piece fits
            display_lines.push_back(remaining);
            break;
        }

        // Break at exact width (like terminal does)
        display_lines.push_back(remaining.substr(0, current_line_width));
        remaining = remaining.substr(current_line_width);

        // Subsequent lines use full terminal width
        current_line_width = terminal_width;
    }

    // Display all lines properly
    for (size_t i = 0; i < display_lines.size(); i++) {
        if (i > 0) {
            // Move to next line for continuation (no prompt)
            std::cout << "\n" << display_lines[i];
        } else {
            // First line (already after prompt)
            std::cout << display_lines[i];
        }
    }

    // Calculate cursor position across the wrapped lines
    // We need to find which display line contains our cursor position

    // First, calculate the visible length up to cursor position
    std::string text_before_cursor = current_line.substr(0, cursor_pos);
    std::string colored_before_cursor = colorize_sql(text_before_cursor);
    size_t visible_chars_before_cursor = visible_length(colored_before_cursor);

    // Now figure out which line and position this corresponds to
    int target_line = 0;
    int target_column = 0;
    size_t chars_counted = 0;

    for (size_t i = 0; i < display_lines.size(); i++) {
        size_t line_visible_length = visible_length(display_lines[i]);

        if (visible_chars_before_cursor <= chars_counted + line_visible_length) {
            target_line = i;
            target_column = visible_chars_before_cursor - chars_counted;
            break;
        }
        chars_counted += line_visible_length;
    }

    // If cursor is beyond all content, put it at end of last line
    if (target_line == display_lines.size() - 1 &&
        target_column > visible_length(display_lines.back())) {
        target_column = visible_length(display_lines.back());
    }

    // Position cursor correctly
    if (target_line > 0) {
        // For multi-line, we need to move to the correct line
        // Move up to first line, then down to target line
        std::cout << "\033[" << target_line << "A"; // Move up to first line
        std::cout << "\033[" << target_line << "B"; // Then down to target line
    }

    // Calculate final column position
    int final_column;
    if (target_line == 0) {
        // First line: column = prompt_length + position_in_line
        final_column = prompt_length + target_column;
    } else {
        // Continuation lines: column = position_in_line (starts at 0)
        final_column = target_column;
    }

    std::cout << "\033[" << final_column << "G";
    std::cout.flush();
}

/*void ESQLShell::redraw_interface() {
    if (is_termux()) return;

    handle_terminal_resize();

    std::string prompt_str = "[" + get_current_time() + "] • " + current_db + "> ";
    int prompt_length = visible_length(prompt_str);

    // Clear entire current input area
    std::cout << "\r\033[K"; // Clear current line

    // Print prompt
    print_prompt();

    if (current_line.empty()) {
        std::cout.flush();
        return;
    }

    std::string colored_line = colorize_sql(current_line);
    int available_width = terminal_width - prompt_length;

    if (available_width < 10) available_width = 10;

    // Calculate how many lines we'll need based on visible length
    size_t total_visible_length = visible_length(colored_line);
    int total_lines = (total_visible_length + available_width - 1) / available_width;

    // Clear any existing lines below
    for (int i = 0; i < total_lines; i++) {
        std::cout << "\n\033[K";
    }

    // Move back to first line
    if (total_lines > 0) {
        std::cout << "\033[" << total_lines << "A";
    }

    // Now display the text with proper wrapping
    std::string remaining = colored_line;
    int line_width = available_width;

    for (int line_num = 0; line_num < total_lines && !remaining.empty(); line_num++) {
        if (line_num > 0) {
            // Move to next line and clear it
            std::cout << "\n\033[K";
            line_width = terminal_width;
        }

        // Display this line's content
        if (remaining.length() <= line_width) {
            std::cout << remaining;
            remaining.clear();
        } else {
            // Break at exact character count (like terminal)
            std::cout << remaining.substr(0, line_width);
            remaining = remaining.substr(line_width);
        }
    }

    // Calculate cursor position using simple math
    std::string text_before_cursor = current_line.substr(0, cursor_pos);
    std::string colored_before_cursor = colorize_sql(text_before_cursor);
    size_t visible_before_cursor = visible_length(colored_before_cursor);

    int cursor_line = visible_before_cursor / available_width;
    int cursor_col = visible_before_cursor % available_width;

    // Position cursor
    if (cursor_line > 0) {
        std::cout << "\033[" << cursor_line << "A"; // Move up
        std::cout << "\033[" << cursor_line << "B"; // Then down (forces positioning)
    }

    // Calculate final column
    int final_col = (cursor_line == 0) ? (prompt_length + cursor_col) : cursor_col;
    std::cout << "\033[" << final_col << "G";

    std::cout.flush();
}*/
/*void ESQLShell::redraw_interface() {
    if (is_termux()) return;

    handle_terminal_resize();

    std::string prompt_str = "[" + get_current_time() + "] • " + current_db + "> ";
    int prompt_length = visible_length(prompt_str);

    // DEBUG
    std::cerr << "[DEBUG] term_width=" << terminal_width
              << ", prompt_len=" << prompt_length
              << ", input_len=" << current_line.length()
              << ", cursor_pos=" << cursor_pos << "\n";

    // Clear everything from cursor down
    std::cout << "\r\033[J"; // Clear from cursor to end of screen

    print_prompt();

    if (current_line.empty()) {
        std::cout.flush();
        return;
    }

    std::string colored_line = colorize_sql(current_line);
    int available_width = terminal_width - prompt_length;

    // Simple character-based wrapping (like terminal does)
    for (size_t i = 0; i < colored_line.length(); i += available_width) {
        if (i > 0) {
            std::cout << "\n"; // Move to next line for continuation
        }
        std::cout << colored_line.substr(i, available_width);
    }

    // Calculate cursor position
    std::string visible_prefix = current_line.substr(0, cursor_pos);
    size_t visible_cursor_pos = visible_length(colorize_sql(visible_prefix));

    int cursor_line = visible_cursor_pos / available_width;
    int cursor_col = visible_cursor_pos % available_width;

    if (cursor_line > 0) {
        std::cout << "\033[" << cursor_line << "A"; // Move up
        std::cout << "\033[" << cursor_line << "B"; // Then down (forces position)
    }

    std::cout << "\033[" << (prompt_length + cursor_col) << "G";
    std::cout.flush();
}*/

/*void ESQLShell::redraw_interface() {
    if (is_termux()) return;

    handle_terminal_resize();

    std::string prompt_str = "[" + get_current_time() + "] • " + current_db + "> ";
    int prompt_length = visible_length(prompt_str);

    // Clear the entire current input area (current line and any continuation lines)
    std::cout << "\r\033[K"; // Clear current line

    // Print prompt
    print_prompt();

    if (current_line.empty()) {
        std::cout.flush();
        return;
    }

    // Colorize the entire input
    std::string colored_line = colorize_sql(current_line);

    // Calculate available width for first line
    int available_width = terminal_width - prompt_length;
    if (available_width < 10) available_width = 10;

    // Break into lines manually (mimicking terminal behavior)
    std::vector<std::string> display_lines;
    std::string remaining = colored_line;
    int line_width = available_width; // First line has prompt

    while (!remaining.empty()) {
        if (remaining.length() <= line_width) {
            display_lines.push_back(remaining);
            break;
        }

        // Break at the exact width (terminal-style, not word-aware)
        display_lines.push_back(remaining.substr(0, line_width));
        remaining = remaining.substr(line_width);

        // Subsequent lines don't have prompt, so full terminal width
        line_width = terminal_width;
    }

    // Display all lines
    for (size_t i = 0; i < display_lines.size(); i++) {
        if (i > 0) {
            // Continuation line - move to next line and print
            std::cout << "\n" << display_lines[i];
        } else {
            // First line - already after prompt
            std::cout << display_lines[i];
        }
    }

    // Calculate cursor position across wrapped lines
    if (display_lines.size() == 1) {
        // Single line - simple positioning
        std::string visible_prefix = current_line.substr(0, cursor_pos);
        size_t visible_cursor_pos = visible_length(colorize_sql(visible_prefix));
        std::cout << "\033[" << (prompt_length + visible_cursor_pos) << "G";
    } else {
        // Multi-line - calculate which line and position
        size_t chars_so_far = 0;
        int target_line = 0;
        size_t pos_in_line = 0;

        // Calculate raw character positions (without colors) for cursor math
        std::string raw_remaining = current_line;
        int current_line_width = available_width;

        for (int i = 0; i < display_lines.size(); i++) {
            size_t line_raw_length = visible_length(display_lines[i]);

            if (cursor_pos <= chars_so_far + line_raw_length) {
                target_line = i;
                pos_in_line = cursor_pos - chars_so_far;
                break;
            }
            chars_so_far += line_raw_length;
            current_line_width = terminal_width; // Subsequent lines use full width
        }

        // If we didn't find the line (cursor at end), use last line
        if (target_line == display_lines.size() - 1 && pos_in_line > visible_length(display_lines.back())) {
            pos_in_line = visible_length(display_lines.back());
        }

        // Position cursor
        if (target_line > 0) {
            // Move to the correct line
            std::cout << "\033[" << target_line << "A"; // Move up to first line
            std::cout << "\033[" << target_line << "B"; // Then down to target line
        }

        // Calculate column position
        int column_pos;
        if (target_line == 0) {
            column_pos = prompt_length + pos_in_line;
        } else {
            column_pos = pos_in_line; // Continuation lines start at column 0
        }

        std::cout << "\033[" << column_pos << "G";
    }

    std::cout.flush();
}*/

/*void ESQLShell::redraw_interface() {
    if (is_termux()) return;

    handle_terminal_resize();

    std::string prompt_str = "[" + get_current_time() + "] • " + current_db + "> ";
    int prompt_length = visible_length(prompt_str);

    // BULLETPROOF CLEARING: Clear from cursor to end of screen
    std::cout << "\r\033[K"; // Clear current line

    // Print prompt
    print_prompt();

    if (current_line.empty()) {
        std::cout.flush();
        return;
    }

    std::string colored_line = colorize_sql(current_line);
    int available_width = terminal_width - prompt_length;

    if (available_width < 10) available_width = 10;

    // Calculate how many lines we'll need
    int total_lines_needed = (visible_length(colored_line) + available_width - 1) / available_width;

    // Clear any existing continuation lines below
    for (int i = 0; i < total_lines_needed; i++) {
        std::cout << "\n\033[K";
    }

    // Move back to the first line
    if (total_lines_needed > 0) {
        std::cout << "\033[" << total_lines_needed << "A";
    }

    // Now print the wrapped content
    std::string remaining = colored_line;
    int current_width = available_width;

    for (int i = 0; i < total_lines_needed && !remaining.empty(); i++) {
        if (i > 0) {
            // Move to next line and clear it
            std::cout << "\n\033[K";
            current_width = terminal_width;
        }

        if (remaining.length() <= current_width) {
            std::cout << remaining;
            remaining.clear();
        } else {
            std::cout << remaining.substr(0, current_width);
            remaining = remaining.substr(current_width);
        }
    }

    // Calculate and set cursor position
    std::string visible_prefix = current_line.substr(0, cursor_pos);
    size_t visible_prefix_length = visible_length(colorize_sql(visible_prefix));

    int cursor_line = visible_prefix_length / available_width;
    int cursor_column = visible_prefix_length % available_width;

    if (cursor_line > 0) {
        cursor_column += prompt_length; // First line has prompt offset
    }

    // Position cursor
    if (cursor_line > 0) {
        std::cout << "\033[" << cursor_line << "A";
        std::cout << "\033[" << cursor_line << "B";
    }
    std::cout << "\033[" << (prompt_length + cursor_column) << "G";

    std::cout.flush();
}*/

/*void ESQLShell::redraw_interface() {
    if (is_termux()) return;

    handle_terminal_resize();

    std::string prompt_str = "[" + get_current_time() + "] • " + current_db + "> ";
    int prompt_length = visible_length(prompt_str);

    // Clear from cursor to end of screen to remove any old continuation lines
    std::cout << "\r\033[K"; // Clear current line

    // Print prompt
    print_prompt();

    if (current_line.empty()) {
        std::cout.flush();
        return;
    }

    std::string colored_line = colorize_sql(current_line);
    int available_width = terminal_width - prompt_length;

    if (available_width < 20) available_width = 20; // Reasonable minimum

    // Simple word-aware line breaking
    std::vector<std::string> lines;
    std::string remaining = colored_line;
    int current_indent = prompt_length;

    while (!remaining.empty()) {
        int max_chars = (lines.empty()) ? available_width : (terminal_width - 2);

        if (remaining.length() <= max_chars) {
            lines.push_back(remaining);
            break;
        }

        // Find break point - prefer breaking at whitespace
        size_t break_pos = max_chars;
        for (size_t i = max_chars; i > max_chars - 10 && i > 0; --i) {
            if (i < remaining.length() && std::isspace(remaining[i])) {
                break_pos = i + 1; // Include the space
                break;
            }
        }

        // Ensure we don't break in the middle of an ANSI escape sequence
        size_t escape_pos = remaining.find("\033[", max_chars - 5);
        if (escape_pos != std::string::npos && escape_pos < max_chars) {
            break_pos = escape_pos; // Break before escape sequence
        }

        lines.push_back(remaining.substr(0, break_pos));
        remaining = remaining.substr(break_pos);

        // Trim leading whitespace from continuation lines
        if (!remaining.empty()) {
            size_t first_non_ws = remaining.find_first_not_of(" \t");
            if (first_non_ws != std::string::npos && first_non_ws > 0) {
                remaining = remaining.substr(first_non_ws);
            }
        }
    }

    // Display all lines
    for (size_t i = 0; i < lines.size(); i++) {
        if (i > 0) {
            // Continuation line
            std::cout << "\n" << std::string(2, ' ') << lines[i];
        } else {
            // First line
            std::cout << lines[i];
        }
    }

    // SIMPLE CURSOR POSITIONING: Always position at end of input
    // This avoids complex multi-line cursor math that causes issues
    if (lines.size() == 1) {
        std::string visible_prefix = current_line.substr(0, cursor_pos);
        size_t visible_cursor_pos = visible_length(colorize_sql(visible_prefix));
        std::cout << "\033[" << (prompt_length + visible_cursor_pos) << "G";
    } else {
        // For multi-line, just position at end - users can use left/right arrows
        std::cout << "\033[" << (2 + visible_length(lines.back())) << "G";
    }

    std::cout.flush();
}*/

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
    bool in_number = false;
    bool in_identifier = false;
    char string_delim = 0;
    std::string current_word;
    std::string current_number;

    // Aggregate functions for special coloring
    static const std::unordered_set<std::string> aggregate_functions = {
        "COUNT", "SUM", "AVG", "MIN", "MAX","DESC","ASC"
    };

    // Operators for gray coloring
    static const std::unordered_set<std::string> operators = {
        "=", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "AND", "OR", "NOT","(",")",","
    };

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (in_string) {
            result += GRAY + std::string(1, c);
            if (c == string_delim) {
                in_string = false;
                result += RESET;
            }
            continue;
        }

        if (in_number) {
            if (std::isdigit(c) || c == '.' || c == '-' || c == '+') {
                current_number += c;
                continue;
            } else {
                result += BLUE + current_number + RESET;
                current_number.clear();
                in_number = false;
                // Continue processing the current character
            }
        }

        // Check for string literals
        if (c == '"' || c == '\'') {
            if (!current_word.empty()) {
                process_word(current_word, result, aggregate_functions, operators);
                current_word.clear();
            }
            in_string = true;
            string_delim = c;
            result += GRAY + std::string(1, c);
            continue;
        }

        // Check for numbers
        if (std::isdigit(c) || (c == '-' && i + 1 < input.size() && std::isdigit(input[i+1]))) {
            if (!current_word.empty()) {
                process_word(current_word, result, aggregate_functions, operators);
                current_word.clear();
            }
            in_number = true;
            current_number += c;
            continue;
        }

        // Check for word boundaries
        if (std::isspace(c) || c == ',' || c == ';' || c == '(' || c == ')' ||
            c == '=' || c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '!' || c == '&' || c == '|') {
            
            if (!current_word.empty()) {
                process_word(current_word, result, aggregate_functions, operators);
                current_word.clear();
            }
            
            // Colorize operators
            if (c == '=' || c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/' ||
                c == '!' || c == '&' || c == '|' || '(' || ')' || ',') {
                result += GRAY + std::string(1, c) + RESET;
            } else {
                result += std::string(1, c);
            }
            continue;
        }

        current_word += c;
    }

    // Process any remaining word
    if (!current_word.empty()) {
        process_word(current_word, result, aggregate_functions, operators);
    }

    // Process any remaining number
    if (!current_number.empty()) {
        result += BLUE + current_number + RESET;
    }

    return result;
}

// Helper function to process words
void ESQLShell::process_word(const std::string& word, std::string& result, 
                           const std::unordered_set<std::string>& aggregate_functions,
                           const std::unordered_set<std::string>& operators) {
    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);

    if (keywords.find(upper_word) != keywords.end()) {
        result += MAGENTA + word + RESET;
    } 
    else if (datatypes.find(upper_word) != datatypes.end()) {
        result += BLUE + word + RESET;
    }
    else if (conditionals.find(upper_word) != conditionals.end()) {
        result += CYAN + word + RESET;
    }
    else if (aggregate_functions.find(upper_word) != aggregate_functions.end()) {
        result += GREEN + word + RESET;
    }
    else if (operators.find(upper_word) != operators.end()) {
        result += GRAY + word + RESET;
    }
    else {
        // Check if it's a quoted identifier (table/column name)
        if (word.size() >= 2 && ((word[0] == '"' && word[word.size()-1] == '"') ||
                                (word[0] == '\'' && word[word.size()-1] == '\''))) {
            result += YELLOW + word + RESET;
        } else {
            // Unquoted identifier (table/column name)
            result += YELLOW + word + RESET;
        }
    }
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
            
            // Check if this was a USE DATABASE command and update prompt
            if (upper_cmd.find("USE ") == 0) {
                // Extract database name from USE command
                size_t use_pos = upper_cmd.find("USE ");
                if (use_pos != std::string::npos) {
                    std::string db_name = command.substr(use_pos + 4);
                    // Remove trailing semicolon if present
                    if (!db_name.empty() && db_name.back() == ';') {
                        db_name.pop_back();
                    }
                    // Trim whitespace
                    db_name.erase(0, db_name.find_first_not_of(" \t"));
                    db_name.erase(db_name.find_last_not_of(" \t") + 1);
                    
                    if (!db_name.empty()) {
                        current_db = db_name;
                    }
                }
            }
            
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
    
    // Special formatting for structure commands
    if (result.columns.size() == 2 && 
        (result.columns[0] == "Property" || result.columns[0] == "Database Structure")) {
        print_structure_results(result, duration);
        return;
    }
    
    // Calculate column widths for regular results
    std::vector<size_t> widths(result.columns.size());
    for (size_t i = 0; i < result.columns.size(); ++i) {
        widths[i] = result.columns[i].length() + 2;
        for (const auto& row : result.rows) {
            if (i < row.size() && row[i].length() + 2 > widths[i]) {
                widths[i] = row[i].length() + 2;
            }
        }
        // Limit maximum width for better display
        if (widths[i] > 50) widths[i] = 50;
    }
    
    // Print header with ASCII styling
    std::cout << (use_colors ? CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n";
    
    std::cout << "|";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << (use_colors ? MAGENTA : "") 
                  << std::left << std::setw(widths[i]) << result.columns[i] 
                  << (use_colors ? CYAN : "") << "|";
    }
    std::cout << (use_colors ? RESET : "") << "\n";
    
    std::cout << (use_colors ? CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n" << (use_colors ? RESET : "");
    
    // Print rows
    for (const auto& row : result.rows) {
        std::cout << (use_colors ? CYAN : "") << "|" << (use_colors ? RESET : "");
        for (size_t i = 0; i < row.size(); ++i) {
            std::string display_value = row[i];
            if (display_value.length() > widths[i] - 2) {
                display_value = display_value.substr(0, widths[i] - 5) + "...";
            }
            std::cout << (use_colors ? YELLOW : "") 
                      << std::left << std::setw(widths[i]) << display_value 
                      << (use_colors ? CYAN : "") << "|" << (use_colors ? RESET : "");
        }
        std::cout << "\n";
    }
    
    // Print footer
    std::cout << (use_colors ? CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n" << (use_colors ? RESET : "");
    
    // Print summary
    std::cout << (use_colors ? GRAY : "") << "> " << result.rows.size() << " row" 
              << (result.rows.size() != 1 ? "s" : "") 
              << " in " << std::fixed << std::setprecision(4) << duration << "s"
              << (use_colors ? RESET : "") << "\n";
}

void ESQLShell::print_structure_results(const ExecutionEngine::ResultSet& result, double duration) {
    std::cout << (use_colors ? CYAN : "") << "==============================================================\n";
    std::cout << (use_colors ? MAGENTA : "") << "               DATABASE STRUCTURE REPORT               \n";
    std::cout << (use_colors ? CYAN : "") << "==============================================================\n" 
              << (use_colors ? RESET : "");
    
    std::string current_section = "";
    
    for (const auto& row : result.rows) {
        if (row.size() < 2) continue;
        
        const std::string& left = row[0];
        const std::string& right = row[1];
        
        // Handle section headers
        if (right.empty() && (left.find("COLUMNS") != std::string::npos || 
                             left.find("CONSTRAINTS") != std::string::npos ||
                             left.find("TABLE DETAILS") != std::string::npos ||
                             left.find("STORAGE INFO") != std::string::npos)) {
            std::cout << "\n" << (use_colors ? GREEN : "") << "=== " << left 
                      << " " << std::string(50 - left.length(), '=') << "\n"
                      << (use_colors ? RESET : "");
            current_section = left;
            continue;
        }
        
        // Handle separators
        if (left == "---" && right == "---") {
            std::cout << (use_colors ? GRAY : "") 
                      << std::string(60, '-') << "\n" 
                      << (use_colors ? RESET : "");
            continue;
        }
        
        // Handle column headers in columns section
        if (current_section.find("COLUMNS") != std::string::npos && 
            left.empty() && right.find("Name | Type") != std::string::npos) {
            std::cout << (use_colors ? CYAN : "") 
                      << std::left << std::setw(56) << right 
                      << (use_colors ? RESET : "") << "\n";
            std::cout << std::string(56, '-') << "\n" 
                      << (use_colors ? RESET : "");
            continue;
        }
        
        // Regular property-value pairs
        if (!left.empty() && !right.empty()) {
            if (left == right) { // This is a section title
                std::cout << (use_colors ? MAGENTA : "") 
                          << std::left << std::setw(56) << left 
                          << (use_colors ? RESET : "") << "\n";
            } else {
                std::cout << (use_colors ? YELLOW : "") << std::left << std::setw(20) << left 
                          << (use_colors ? CYAN : "") << " : " 
                          << (use_colors ? GRAY : "") << std::left << std::setw(33) << right 
                          << (use_colors ? RESET : "") << "\n";
            }
        } else if (left.empty() && !right.empty()) {
            // Indented information (like column details)
            std::cout << "   " 
                      << (use_colors ? BLUE : "") << std::left << std::setw(53) << right 
                      << (use_colors ? RESET : "") << "\n";
        }
    }
    
    std::cout << (use_colors ? CYAN : "") << std::string(60, '=') << "\n" 
              << (use_colors ? RESET : "");
    
    // Print summary
    std::cout << (use_colors ? GRAY : "") << "> Report generated in " 
              << std::fixed << std::setprecision(4) << duration << "s"
              << (use_colors ? RESET : "") << "\n";
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
