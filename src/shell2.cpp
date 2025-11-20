#include "shell2.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <thread>
#include <cstring>
#include <regex>
#include <wchar.h>
#include <locale>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#endif

// SQL elements for syntax highlighting with enhanced formatting
const std::unordered_set<std::string> ESQLShell::keywords = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DATABASE", "DROP", "ALTER", "ADD", "RENAME",
    "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP", "DISTINCT", "DATABASES","BY","ORDER","GROUP","HAVING","BULK","ROW","TABLES","STRUCTURE"
};

const std::unordered_set<std::string> ESQLShell::datatypes = {
    "INT", "INTEGER", "FLOAT", "TEXT", "STRING", "BOOL", "BOLEAN", "VARCHAR", "DATE", "DATETIME", "UUID"
};

const std::unordered_set<std::string> ESQLShell::constraints = {
    "PRIMARY_KEY", "UNIQUE", "DEFAULT", "AUTO_INCREAMENT", "CHECK", "NOT_NULL", "FOREIGN_KEY", "REFERENCES"
};

const std::unordered_set<std::string> ESQLShell::conditionals = {
    "AND", "OR", "NOT", "NULL", "IS", "LIKE", "IN", "BETWEEN", "OFFSET", "LIMIT","AS","TO","TRUE","FALSE", "GENERATE_UUID", "GENERATE_DATE", "GENERATE_DATE_TIME", "CASE", "WHEN", "THEN", "ELSE"
};

// ------------------------ Construction / terminal ------------------------

ESQLShell::ESQLShell(Database& db) : db(db), current_db("default") {
    current_platform = detect_platform();
    get_terminal_size();
    // compute banner_lines based on your ascii art + header lines
    banner_lines = std::max((int)load_ascii_art().size(), 6) + 4; // approximate
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
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_iflag &= ~(IXON | ICRNL);
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
        terminal_width = 80;
    }
    #else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        terminal_width = ws.ws_col;
    } else {
        const char* columns = getenv("COLUMNS");
        if (columns) terminal_width = std::atoi(columns);
        else terminal_width = 80;
    }
    #endif
    if (terminal_width < 40) terminal_width = 40;
}

// ------------------------ Input reading (tiny) ------------------------

int ESQLShell::read_key() {
    #ifdef _WIN32
    if (_kbhit()) {
        int ch = _getch();
        if (ch == 0 || ch == 224) {
            switch (_getch()) {
                case 72: return 1002;
                case 80: return 1003;
                case 75: return 1000;
                case 77: return 1001;
            }
        }
        return ch;
    }
    return -1;
    #else
    char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    if (n != 1) return -1;
    if (c == '\033') {
        char seq[2];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return '\033';
        if (seq[0] == '[') {
            if (read(STDIN_FILENO, &seq[1], 1) != 1) return '\033';
            switch (seq[1]) {
                case 'A': return 1002;
                case 'B': return 1003;
                case 'C': return 1001;
                case 'D': return 1000;
            }
        }
        return '\033';
    }
    return (unsigned char)c;
    #endif
}

// ------------------------ High level run loops ------------------------

void ESQLShell::run() {
    if (is_termux()) run_termux();
    else run_linux();
}

void ESQLShell::run_linux() {
    refresh_display(); // initial
    while (true) {
        int ch = read_key();
        if (ch == -1) continue;
        switch (ch) {
            case '\n': case '\r':
                handle_enter();
                break;
            case 127: case 8:
                delete_char();
                break;
            case '\t':
                handle_tab_completion();
                break;
            case 1000:
                move_cursor_left();
                break;
            case 1001:
                move_cursor_right();
                break;
            case 1002:
                navigate_history_up();
                break;
            case 1003:
                navigate_history_down();
                break;
            case 4: // ctrl-d
                std::cout << "\n";
                return;
            case 18: // ctrl-r - terminal resize
                handle_terminal_resize();
                break;
            default:
                if (ch >= 32) {
                    insert_char(static_cast<char>(ch));
                }
                break;
        }
    }
}

void ESQLShell::run_termux() {
    if (use_colors) std::cout << BLUE << "Termux mode activated\n\n" << RESET;
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

// ------------------------ Cache Management ------------------------

void ESQLShell::invalidate_cache() {
    render_cache.valid = false;
}

bool ESQLShell::use_cached_render(int prompt_width) {
    if (!render_cache.valid) return false;
    if (render_cache.cached_terminal_width != terminal_width) return false;
    if (render_cache.cached_prompt_width != prompt_width) return false;
    
    // Generate simple hash of current input for cache validation
    std::string current_hash = current_line + "|" + std::to_string(cursor_pos);
    if (render_cache.input_hash != current_hash) return false;
    
    return true;
}

// ------------------------ New cell engine: building cells and attaching ANSI prefixes ------------------------

std::vector<ESQLShell::Cell> ESQLShell::utf8_to_base_cells(const std::string& s) {
    std::vector<Cell> out;
    const char* p = s.c_str();
    size_t len = s.size();
    size_t i = 0;
    std::setlocale(LC_ALL, "en_US.UTF-8");
    while (i < len) {
        wchar_t wc;
        int consumed = mbtowc(&wc, p + i, len - i);
        if (consumed < 1) {
            // fallback single byte
            Cell c;
            c.bytes = std::string(1, s[i]);
            c.width = 1;
            c.prefix.clear();
            out.push_back(c);
            i++;
            continue;
        }
        int w = wcwidth(wc);
        if (w < 0) w = 1;
        Cell c;
        c.bytes.assign(p + i, consumed);
        c.width = w;
        c.prefix.clear();
        out.push_back(c);
        i += consumed;
    }
    return out;
}

/*
 Simplified ANSI prefix attachment - more robust approach
*/
void ESQLShell::attach_ansi_prefixes_from_colorized(const std::string& colorized) {
    // Clear previous prefixes
    for (auto &c : cell_buffer) c.prefix.clear();

    std::string active_prefix;
    size_t cell_index = 0;
    size_t color_index = 0;
    const size_t color_len = colorized.size();

    while (color_index < color_len && cell_index < cell_buffer.size()) {
        if (colorized[color_index] == '\033') {
            // Parse escape sequence
            size_t seq_start = color_index;
            color_index++; // Skip ESC
            
            if (color_index < color_len && colorized[color_index] == '[') {
                color_index++;
                // Read until command letter
                while (color_index < color_len && 
                       !((colorized[color_index] >= 'A' && colorized[color_index] <= 'Z') ||
                         (colorized[color_index] >= 'a' && colorized[color_index] <= 'z'))) {
                    color_index++;
                }
                if (color_index < color_len) {
                    color_index++; // Include final letter
                    std::string seq = colorized.substr(seq_start, color_index - seq_start);
                    
                    if (seq == RESET) {
                        active_prefix.clear();
                    } else {
                        active_prefix = seq; // Replace rather than append for simplicity
                    }
                }
            }
            continue;
        }

        // Regular character - match with cell buffer
        const std::string& cell_bytes = cell_buffer[cell_index].bytes;
        if (color_index + cell_bytes.size() <= color_len && 
            colorized.compare(color_index, cell_bytes.size(), cell_bytes) == 0) {
            cell_buffer[cell_index].prefix = active_prefix;
            color_index += cell_bytes.size();
            cell_index++;
        } else {
            // Fallback: advance one byte in colorized and one cell
            if (cell_index < cell_buffer.size()) {
                cell_buffer[cell_index].prefix = active_prefix;
                cell_index++;
            }
            color_index++;
        }
    }

    // Ensure remaining cells have the active prefix
    for (; cell_index < cell_buffer.size(); cell_index++) {
        cell_buffer[cell_index].prefix = active_prefix;
    }
}

// Build cell_buffer from current_line and attach prefixes using colorize_sql
void ESQLShell::rebuild_cells_from_input(const std::string& input) {
    cell_buffer.clear();
    if (input.empty()) return;
    cell_buffer = utf8_to_base_cells(input);
    std::string colored = colorize_sql(input);
    attach_ansi_prefixes_from_colorized(colored);
}

// ------------------------ wrapping / cursor mapping ------------------------

std::vector<std::vector<ESQLShell::Cell>> ESQLShell::wrap_cells_for_prompt(int prompt_width) {
    // Use cache if available
    if (render_cache.valid && render_cache.cached_terminal_width == terminal_width && 
        render_cache.cached_prompt_width == prompt_width) {
        return render_cache.wrapped_rows;
    }

    std::vector<std::vector<Cell>> rows;
    if (cell_buffer.empty()) {
        rows.push_back({});
        return rows;
    }

    int first_limit = terminal_width - prompt_width;
    if (first_limit < 10) first_limit = terminal_width - 10;
    int cont_limit = terminal_width - 4; // space for " -> " (4 columns)
    if (cont_limit < 10) cont_limit = terminal_width - 10;

    std::vector<Cell> current_row;
    int cur_width = 0;
    int limit = first_limit;
    
    for (const auto &c : cell_buffer) {
        if (!current_row.empty() && cur_width + c.width > limit) {
            rows.push_back(current_row);
            current_row.clear();
            cur_width = 0;
            limit = cont_limit;
        }
        current_row.push_back(c);
        cur_width += c.width;
    }
    
    if (!current_row.empty()) {
        rows.push_back(current_row);
    }

    // Update cache
    render_cache.wrapped_rows = rows;
    render_cache.cached_terminal_width = terminal_width;
    render_cache.cached_prompt_width = prompt_width;
    render_cache.input_hash = current_line + "|" + std::to_string(cursor_pos);
    render_cache.valid = true;

    return rows;
}

ESQLShell::CursorPos ESQLShell::cursor_position_from_byte_offset(size_t byte_offset, int prompt_width) {
    // Map byte offset in current_line -> cell index -> (row, col)
    
    // First find cell index that contains or is after the byte offset
    size_t accum = 0;
    size_t cell_idx = 0;
    
    // FIXED: Use >= instead of > to correctly handle cursor positioning
    while (cell_idx < cell_buffer.size()) {
        size_t blen = cell_buffer[cell_idx].bytes.size();
        if (accum + blen > byte_offset) break; // cursor is before this cell
        accum += blen;
        cell_idx++;
    }
    
    // cell_idx now points to the cell that the cursor is before
    // If we reached the end, cursor is after last cell
    if (cell_idx >= cell_buffer.size()) {
        cell_idx = cell_buffer.size();
    }

    // Compute wrapped rows
    auto rows = wrap_cells_for_prompt(prompt_width);
    
    // Find which row and column contains our cell index
    size_t cells_seen = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        int col = 0;
        for (size_t ci = 0; ci < rows[r].size(); ++ci) {
            if (cells_seen == cell_idx) {
                return {(int)r, col};
            }
            col += rows[r][ci].width;
            cells_seen++;
        }
        // Check if cursor is at end of this row
        if (cells_seen == cell_idx) {
            return {(int)r, col};
        }
    }
    
    // Beyond end - place at end of last row
    if (!rows.empty()) {
        int last_col = 0;
        for (const auto &c : rows.back()) last_col += c.width;
        return {(int)rows.size() - 1, last_col};
    }
    
    return {0, 0};
}

size_t ESQLShell::byte_offset_at_cell_index(size_t cell_index) const {
    if (cell_index >= cell_buffer.size()) {
        // Return total length if beyond end
        size_t total = 0;
        for (const auto& cell : cell_buffer) total += cell.bytes.size();
        return total;
    }
    
    size_t off = 0;
    for (size_t i = 0; i < cell_index; ++i) {
        off += cell_buffer[i].bytes.size();
    }
    return off;
}

// ------------------------ Rendering ------------------------

std::string ESQLShell::build_prompt() const {
    // Format: [HH:MM] • current_db > 
    std::string tm = get_current_time();
    std::string prompt;
    if (use_colors) {
        prompt = std::string(YELLOW) + "[" + tm + "] " + RESET +
                 std::string(GREEN) + "• " + RESET +
                 std::string(GRAY) + current_db + RESET +
                 "> ";
    } else {
        prompt = "[" + tm + "] • " + current_db + "> ";
    }
    return prompt;
}

void ESQLShell::render_input_full() {
    if (is_termux()) return; // termux uses line mode
    get_terminal_size();

    // Rebuild cell buffer from current_line (and attach ANSI prefixes)
    rebuild_cells_from_input(current_line);

    // Build prompt and compute prompt width (visual)
    std::string prompt = build_prompt();
    int prompt_width = 0;
    // compute visual width of prompt (strip ANSI)
    std::string stripped_prompt = strip_ansi_simple(prompt);
    prompt_width = display_width(stripped_prompt);

    // Wrap cells
    auto rows = wrap_cells_for_prompt(prompt_width);

    // Clear screen & banner
    std::cout << "\033[2J\033[H";

    print_banner();

    // Print prompt + first row
    std::cout << prompt;
    // Print first line cells
    if (!rows.empty()) {
        std::string last_prefix;
        for (const auto &cell : rows[0]) {
            // print new prefix if changed
            if (cell.prefix != last_prefix) {
                if (!cell.prefix.empty()) std::cout << cell.prefix;
                else std::cout << RESET;
                last_prefix = cell.prefix;
            }
            std::cout << cell.bytes;
        }
        // terminate coloring for first line
        if (!last_prefix.empty()) std::cout << RESET;
    }

    // continuation rows
    for (size_t r = 1; r < rows.size(); ++r) {
        std::cout << "\r\n";
        if (use_colors) std::cout << GRAY << " -> " << RESET;
        else std::cout << " -> ";
        std::string last_prefix;
        for (const auto &cell : rows[r]) {
            if (cell.prefix != last_prefix) {
                if (!cell.prefix.empty()) std::cout << cell.prefix;
                else std::cout << RESET;
                last_prefix = cell.prefix;
            }
            std::cout << cell.bytes;
        }
        if (!last_prefix.empty()) std::cout << RESET;
    }

    // Position cursor
    CursorPos cur = cursor_position_from_byte_offset(cursor_pos, prompt_width);

    // Move cursor to correct position
    int move_up = static_cast<int>(rows.size() - 1) - cur.row;
    if (move_up > 0) {
        std::cout << "\033[" << move_up << "A";
    }
    
    // Set horizontal position
    if (cur.row == 0) {
        int col = prompt_width + cur.col;
        if (col < 1) col = 1;
        std::cout << "\033[" << col << "G";
    } else {
        int cont_prefix_width = 4; // " -> "
        int col = cont_prefix_width + cur.col;
        if (col < 1) col = 1;
        std::cout << "\033[" << col << "G";
    }

    std::cout.flush();
}

void ESQLShell::refresh_display() {
    if (is_termux()) return;
    render_input_full();
}

void ESQLShell::redraw_input_enhanced() {
    refresh_display();
}

void ESQLShell::redraw_input() {
    refresh_display();
}

void ESQLShell::handle_terminal_resize() {
    get_terminal_size();
    invalidate_cache();
    refresh_display();
}

std::string ESQLShell::strip_ansi_simple(const std::string& s) {
    std::string out;
    size_t i = 0;
    size_t n = s.size();
    while (i < n) {
        unsigned char ch = s[i];
        if (ch == 0x1b) {
            i++;
            if (i < n && s[i] == '[') {
                i++;
                while (i < n && !((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z'))) i++;
                if (i < n) i++;
            }
        } else {
            out.push_back(s[i]);
            i++;
        }
    }
    return out;
}

int ESQLShell::display_width(const std::string& s) {
    // naive width: sum of wcwidth of each wchar
    const char* p = s.c_str();
    size_t len = s.size();
    size_t i = 0;
    std::setlocale(LC_ALL, "en_US.UTF-8");
    int wsum = 0;
    while (i < len) {
        wchar_t wc;
        int consumed = mbtowc(&wc, p + i, len - i);
        if (consumed < 1) {
            wsum += 1;
            i++;
            continue;
        }
        int w = wcwidth(wc);
        if (w < 0) w = 1;
        wsum += w;
        i += consumed;
    }
    return wsum;
}

// ------------------------ Editing / navigation (hook into new engine) ------------------------

void ESQLShell::insert_char(char c) {
    // insert at byte cursor
    current_line.insert(cursor_pos, 1, c);
    cursor_pos++;
    invalidate_cache();
    refresh_display();
}

void ESQLShell::delete_char() {
    if (cursor_pos == 0 || current_line.empty()) return;
    // remove previous UTF-8 glyph safely: find previous byte boundary
    size_t pos = cursor_pos;
    // back up one UTF-8 character
    do {
        pos--;
    } while (pos > 0 && ( (current_line[pos] & 0xC0) == 0x80 )); // continuation bytes
    current_line.erase(pos, cursor_pos - pos);
    cursor_pos = pos;
    invalidate_cache();
    refresh_display();
}

void ESQLShell::move_cursor_left() {
    if (cursor_pos == 0) return;
    // move to previous char start
    size_t pos = cursor_pos;
    do {
        pos--;
    } while (pos > 0 && ( (current_line[pos] & 0xC0) == 0x80 ));
    cursor_pos = pos;
    invalidate_cache();
    refresh_display();
}

void ESQLShell::move_cursor_right() {
    if (cursor_pos >= current_line.size()) return;
    // move to next char boundary
    size_t pos = cursor_pos;
    unsigned char lead = current_line[pos];
    size_t char_len = 1;
    if ((lead & 0x80) == 0) char_len = 1;
    else if ((lead & 0xE0) == 0xC0) char_len = 2;
    else if ((lead & 0xF0) == 0xE0) char_len = 3;
    else if ((lead & 0xF8) == 0xF0) char_len = 4;
    else char_len = 1;
    cursor_pos = std::min(current_line.size(), pos + char_len);
    invalidate_cache();
    refresh_display();
}

void ESQLShell::navigate_history_up() {
    if (command_history.empty()) return;
    if (history_index == -1) history_index = static_cast<int>(command_history.size()) - 1;
    else if (history_index > 0) history_index--;
    current_line = command_history[history_index];
    cursor_pos = current_line.size();
    invalidate_cache();
    refresh_display();
}

void ESQLShell::navigate_history_down() {
    if (command_history.empty()) return;
    if (history_index < static_cast<int>(command_history.size()) - 1) {
        history_index++;
        current_line = command_history[history_index];
    } else {
        history_index = static_cast<int>(command_history.size());
        current_line.clear();
    }
    cursor_pos = current_line.size();
    invalidate_cache();
    refresh_display();
}

// ------------------------ Completion, enter, history ------------------------

void ESQLShell::handle_tab_completion() {
    auto suggestions = get_completion_suggestion(current_line);
    if (suggestions.empty()) return;
    if (suggestions.size() == 1) {
        current_line = suggestions[0];
        cursor_pos = current_line.size();
    } else {
        // show options then redraw
        std::cout << "\n";
        for (const auto &s : suggestions) {
            std::cout << " " << colorize_sql(s) << "\n";
        }
        std::cout << "\n";
    }
    invalidate_cache();
    refresh_display();
}

std::vector<std::string> ESQLShell::get_completion_suggestion(const std::string& input) {
    std::vector<std::string> suggestions;
    std::string last_word;
    size_t last_space = input.find_last_of(" \t\n,;()");
    if (last_space == std::string::npos) last_word = input;
    else last_word = input.substr(last_space + 1);
    std::string upper_last_word = last_word;
    std::transform(upper_last_word.begin(), upper_last_word.end(), upper_last_word.begin(), ::toupper);

    for (const auto& kw : keywords) {
        if (kw.find(upper_last_word) == 0) {
            std::string completion = (last_space == std::string::npos ? std::string() : input.substr(0, last_space + 1)) + kw;
            suggestions.push_back(completion);
        }
    }
    for (const auto& dt : datatypes) {
        if (dt.find(upper_last_word) == 0) {
            std::string completion = (last_space == std::string::npos ? std::string() : input.substr(0, last_space + 1)) + dt;
            suggestions.push_back(completion);
        }
    }
    for (const auto& cn : constraints) {
        if (cn.find(upper_last_word) == 0) {
            std::string completion = (last_space == std::string::npos ? std::string() : input.substr(0, last_space + 1)) + cn;
            suggestions.push_back(completion);
        }
    }
    for (const auto& fn : conditionals) {
        if (fn.find(upper_last_word) == 0) {
            std::string completion = (last_space == std::string::npos ? std::string() : input.substr(0, last_space + 1)) + fn;
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
    invalidate_cache();
    refresh_display();
}

void ESQLShell::add_to_history(const std::string& command) {
    if (!command.empty() && (command_history.empty() || command_history.back() != command)) {
        command_history.push_back(command);
    }
}

// ------------------------ Execute & output formatting (kept from your original) ------------------------

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

            if (upper_cmd.find("USE ") == 0) {
                size_t use_pos = upper_cmd.find("USE ");
                if (use_pos != std::string::npos) {
                    std::string db_name = command.substr(use_pos + 4);
                    if (!db_name.empty() && db_name.back() == ';') db_name.pop_back();
                    db_name.erase(0, db_name.find_first_not_of(" \t"));
                    db_name.erase(db_name.find_last_not_of(" \t") + 1);
                    if (!db_name.empty()) current_db = db_name;
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

// ------------------------ Results printing (kept from original) ------------------------

void ESQLShell::print_results(const ExecutionEngine::ResultSet& result, double duration) {
    if (result.columns.empty()) {
        std::cout << (use_colors ? GREEN : "") << "Query executed successfully.\n" << (use_colors ? RESET : "");
        return;
    }
    if (result.columns.size() == 2 &&
        (result.columns[0] == "Property" || result.columns[0] == "Database Structure")) {
        print_structure_results(result, duration);
        return;
    }

    std::vector<size_t> widths(result.columns.size());
    for (size_t i = 0; i < result.columns.size(); ++i) {
        widths[i] = result.columns[i].length() + 2;
        for (const auto& row : result.rows) {
            if (i < row.size() && row[i].length() + 2 > widths[i]) {
                widths[i] = row[i].length() + 2;
            }
        }
        if (widths[i] > 50) widths[i] = 50;
    }

    std::cout << (use_colors ? CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n";

    std::cout << "|";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << (use_colors ? MAGENTA : "") << std::left << std::setw(widths[i]) << result.columns[i]
                  << (use_colors ? CYAN : "") << "|";
    }
    std::cout << (use_colors ? RESET : "") << "\n";

    std::cout << (use_colors ? CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n" << (use_colors ? RESET : "");

    for (const auto& row : result.rows) {
        std::cout << (use_colors ? CYAN : "") << "|" << (use_colors ? RESET : "");
        for (size_t i = 0; i < row.size(); ++i) {
            std::string display_value = row[i];
            if (display_value.length() > widths[i] - 2) {
                display_value = display_value.substr(0, widths[i] - 5) + "...";
            }
            std::cout << (use_colors ? YELLOW : "") << std::left << std::setw(widths[i]) << display_value
                      << (use_colors ? CYAN : "") << "|" << (use_colors ? RESET : "");
        }
        std::cout << "\n";
    }

    std::cout << (use_colors ? CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n" << (use_colors ? RESET : "");

    std::cout << (use_colors ? GRAY : "") << "> " << result.rows.size() << " row"
              << (result.rows.size() != 1 ? "s" : "") << " in " << std::fixed << std::setprecision(4)
              << duration << "s" << (use_colors ? RESET : "") << "\n";
}

void ESQLShell::print_structure_results(const ExecutionEngine::ResultSet& result, double duration) {
    std::cout << (use_colors ? CYAN : "") << "==============================================================\n";
    std::cout << (use_colors ? MAGENTA : "") << "               DATABASE STRUCTURE REPORT               \n";
    std::cout << (use_colors ? CYAN : "") << "==============================================================\n" << (use_colors ? RESET : "");

    std::string current_section = "";

    for (const auto& row : result.rows) {
        if (row.size() < 2) continue;
        const std::string& left = row[0];
        const std::string& right = row[1];

        if (right.empty() && (left.find("COLUMNS") != std::string::npos ||
                              left.find("CONSTRAINTS") != std::string::npos ||
                              left.find("TABLE DETAILS") != std::string::npos ||
                              left.find("STORAGE INFO") != std::string::npos)) {
            std::cout << "\n" << (use_colors ? GREEN : "") << "=== " << left << " " << std::string(50 - left.length(), '=') << "\n" << (use_colors ? RESET : "");
            current_section = left;
            continue;
        }
        if (left == "---" && right == "---") {
            std::cout << (use_colors ? GRAY : "") << std::string(60, '-') << "\n" << (use_colors ? RESET : "");
            continue;
        }
        if (current_section.find("COLUMNS") != std::string::npos && left.empty() && right.find("Name | Type") != std::string::npos) {
            std::cout << (use_colors ? CYAN : "") << std::left << std::setw(56) << right << (use_colors ? RESET : "") << "\n";
            std::cout << std::string(56, '-') << "\n" << (use_colors ? RESET : "");
            continue;
        }
        if (!left.empty() && !right.empty()) {
            if (left == right) {
                std::cout << (use_colors ? MAGENTA : "") << std::left << std::setw(56) << left << (use_colors ? RESET : "") << "\n";
            } else {
                std::cout << (use_colors ? YELLOW : "") << std::left << std::setw(20) << left << (use_colors ? CYAN : "") << " : " << (use_colors ? GRAY : "") << std::left << std::setw(33) << right << (use_colors ? RESET : "") << "\n";
            }
        } else if (left.empty() && !right.empty()) {
            std::cout << "   " << (use_colors ? BLUE : "") << std::left << std::setw(53) << right << (use_colors ? RESET : "") << "\n";
        }
    }

    std::cout << (use_colors ? CYAN : "") << std::string(60, '=') << "\n" << (use_colors ? RESET : "");
    std::cout << (use_colors ? GRAY : "") << "> Report generated in " << std::fixed << std::setprecision(4) << duration << "s" << (use_colors ? RESET : "") << "\n";
}

// ------------------------ Banner / prompt / colorize (kept from original) ------------------------

std::vector<std::string> ESQLShell::load_ascii_art() {
    return {
        color_line("                    ...  .:....", YELLOW),
        color_line("         ..,l:c,'.....,;;;,.. ", YELLOW),
        color_line("      ..ccoo;'...      .',ll::..", YELLOW),
        color_line("     .,xxodc..          .,:dodl,.", YELLOW),
        color_line("   ..clkkoo,             .;oddxl;'.", YELLOW),
        color_line("  .,:ddxxkd,.            .cddddocc:.", YELLOW),
        color_line("  ';ooxllkx:             .lxodoxll:.", YELLOW),
        color_line(" ..cldolclxl.  . ..     .,xooldooc:'", YELLOW),
        color_line(" .,;cllc:;cdd, .:xkl,..,ddc::looolc'", YELLOW),
        color_line("  ';loll:,,,:xl.'do:..ld:;,;cloooc'", YELLOW),
        color_line("   .,coooc:,,;lododxddc,,,;coool;.", YELLOW),
        color_line("     ..;clll:;,,lc:::,,;:cllc:,.", YELLOW),
        color_line("        ..,;::c;;:,,c,::cc,...", YELLOW),
        color_line("         ....',,::,:cc:,'...", YELLOW),
        color_line("         ;,',;::c;,,ccc::;'.", YELLOW),
        color_line("       ,':;,;:cccll:ccl:;''''..", YELLOW),
        color_line("       .,,':,;ccl:c;:::,,;..,;", YELLOW),
        color_line("       ....',,;c:;;;:;;;;.'.'.", YELLOW),
        color_line("         ....','.......''....", YELLOW)
    };
}

std::string ESQLShell::color_line(const std::string& line, const char* color) {
    return std::string(color) + line + RESET;
}

void ESQLShell::print_banner() {
    if (use_colors) {
        auto phoenix_art = load_ascii_art();
        size_t max_lines = std::max(phoenix_art.size(), size_t(6));
        for (size_t i = 0; i < max_lines; ++i) {
            if (i < phoenix_art.size()) std::cout << phoenix_art[i];
            else std::cout << std::string(40, ' ');
            switch (i) {
                case 0: std::cout << "   " << GRAY << "███████╗███████╗ ██████╗ ██╗   " << RESET; break;
                case 1: std::cout << "   " << GRAY << "██╔════╝██╔════╝██╔═══██╗██║   " << RESET; break;
                case 2: std::cout << "   " << GRAY << "█████╗  ███████╗██║   ██║██║   " << RESET; break;
                case 3: std::cout << "   " << GRAY << "██╔══╝  ╚════██║██║   ██║██║   " << RESET; break;
                case 4: std::cout << "   " << GRAY << "███████╗███████║╚██████╔╝███████╗" << RESET; break;
                case 5: std::cout << "   " << GRAY << "╚══════╝╚══════╝ ╚═════╝ ╚══════╝" << RESET; break;
                default: std::cout << "                                      "; break;
            }
            std::cout << "\n";
        }
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
    } else {
        std::cout << "ESQL Shell - connected to: " << current_db << "\n\n";
    }
}

void ESQLShell::print_prompt() {
    std::cout << build_prompt();
}

std::string ESQLShell::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M");
    return ss.str();
}

// ------------------------ colorize / tokenization (enhanced with bold formatting) ------------------------

std::string ESQLShell::colorize_sql(const std::string& input) {
    std::string result;
    bool in_string = false;
    bool in_number = false;
    char string_delim = 0;
    std::string current_word;
    std::string current_number;

    static const std::unordered_set<std::string> aggregate_functions = {
        "COUNT", "SUM", "AVG", "MIN", "MAX","DESC","ASC"
    };

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
            }
        }

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

        if (std::isdigit(c) || (c == '-' && i + 1 < input.size() && std::isdigit(input[i+1]))) {
            if (!current_word.empty()) {
                process_word(current_word, result, aggregate_functions, operators);
                current_word.clear();
            }
            in_number = true;
            current_number += c;
            continue;
        }

        if (std::isspace((unsigned char)c) || c == ',' || c == ';' || c == '(' || c == ')' ||
            c == '=' || c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '!' || c == '&' || c == '|') {

            if (!current_word.empty()) {
                process_word(current_word, result, aggregate_functions, operators);
                current_word.clear();
            }

            if (c == '=' || c == '<' || c == '>' || c == '+' || c == '-' || c == '*' || c == '/' ||
                c == '!' || c == '&' || c == '|' || c == '(' || c == ')' || c == ',') {
                result += GRAY + std::string(1, c) + RESET;
            } else {
                result += std::string(1, c);
            }
            continue;
        }

        current_word += c;
    }

    if (!current_word.empty()) {
        process_word(current_word, result, aggregate_functions, operators);
    }

    if (!current_number.empty()) {
        result += BLUE + current_number + RESET;
    }

    return result;
}

void ESQLShell::process_word(const std::string& word, std::string& result,
                             const std::unordered_set<std::string>& aggregate_functions,
                             const std::unordered_set<std::string>& operators) {
    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);

    if (keywords.find(upper_word) != keywords.end()) {
        result += MAGENTA + word + RESET;
    }
    else if (datatypes.find(upper_word) != datatypes.end()) {
        // ENHANCED: Data types in bold blue
        result += BOLD_BLUE + word + RESET;
    }
    else if (constraints.find(upper_word) != constraints.end()) {
        // ENHANCED: Constraints in bold cyan  
        result += BOLD_CYAN + word + RESET;
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
        if (word.size() >= 2 && ((word[0] == '"' && word[word.size()-1] == '"') ||
                                (word[0] == '\'' && word[word.size()-1] == '\''))) {
            result += YELLOW + word + RESET;
        } else {
            result += YELLOW + word + RESET;
        }
    }
}

// ------------------------ Help / utility / show_help ------------------------

void ESQLShell::show_help() {
    std::cout << "\n" << (use_colors ? CYAN : "") << "Available commands:" << (use_colors ? RESET : "") << "\n";
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
        std::cout << "  " << (use_colors ? MAGENTA : "") << cmd << (use_colors ? GREEN : "") << " - " << desc << (use_colors ? RESET : "") << "\n";
    }
    std::cout << "\n";
}

void ESQLShell::setCurrentDatabase(const std::string& db_name) {
    current_db = db_name;
}

void ESQLShell::clear_screen() {
    std::cout << "\033[2J\033[H";
}

bool ESQLShell::is_termux() const {
    return current_platform == Platform::Termux;
}
