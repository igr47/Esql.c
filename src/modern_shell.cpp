#include "modern_shell.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/select.h>
#endif

ModernShell::ModernShell(Database& db) 
    : db_(db), current_db_("default") {
    
    // Initialize terminal
    terminal_.enable_raw_mode();
    terminal_.get_terminal_size(terminal_width_, terminal_height_);
    
    highlighter_.enable_colors(use_colors_);
    highlighter_.set_current_database(current_db_);

    // Initial prompt positions is below the banner
    prompt_row_ = 19;
    prompt_col_ = 1;
}

ModernShell::~ModernShell() {
    terminal_.disable_raw_mode();
    // Reset terminal state
    std::cout << "\033[0m\r\n" << std::endl;
}

void ModernShell::run() {
    if (terminal_.is_termux()) {
        run_termux_fallback();
    } else {
        run_interactive();
    }
}

void ModernShell::run_interactive() {
    clear_screen();
    print_banner();
    
    // Initialize state
    current_input_.clear();
    cursor_pos_ = 0;
    last_rendered_input_.clear();
    last_cursor_pos_ = 0;

    // Position for first prompt
    move_to_prompt_position();
    print_prompt();
    update_prompt_position(); // Store hwre we placed the prompt
    
    refresh_display(true); // Force initial display

    while (!should_exit_) {
        char c;
        ssize_t n = read(STDIN_FILENO, &c, 1);

        if (n != 1) continue;

        // Check for resize first
        if (c == '\x1B') {
            if (handle_possible_resize()) {
                continue;
            }
        }

        esql::KeyCode key = convert_char_to_keycode(c);
        
        switch (key) {
            case esql::KeyCode::Enter:
                handle_enter();
                break;
            case esql::KeyCode::Backspace:
                handle_backspace();
                break;
            case esql::KeyCode::Tab:
                handle_tab();
                break;
            case esql::KeyCode::Character:
                handle_character(c);
                break;
            case esql::KeyCode::Up:
            case esql::KeyCode::Down:
            case esql::KeyCode::Left:
            case esql::KeyCode::Right:
            case esql::KeyCode::Home:
            case esql::KeyCode::End:
                handle_navigation(key);
                break;
            case esql::KeyCode::CtrlD:
                should_exit_ = true;
                break;
            case esql::KeyCode::CtrlL:
                clear_screen();
                print_banner();
                refresh_display(true);
                break;
            case esql::KeyCode::Escape:
                // Ignore standalone escape
                break;
            default:
                break;
        }
    }

    std::cout << "\n";
}

void ModernShell::update_prompt_position() {
    terminal_.get_cursor_position(prompt_row_, prompt_col_);

    // Adjust for the fact that we  want the position BEFORE the prompt was printed
    // Sice get_cursor get_cursor_position returns where we are AFTER printing

    prompt_col_ = 1;
}

void ModernShell::move_to_prompt_position() {
    terminal_.move_cursor(prompt_row_, prompt_col_);
}

void ModernShell::update_screen() {
    // Move to where the prompt is currently located
    move_to_prompt_position();
    
    std::string current_prompt = build_prompt();
    std::string colored_input = highlighter_.highlight(current_input_);
    
    // Clear from prompt position to end of line
    std::cout << "\033[K" << current_prompt;
    
    // Output colored input
    std::cout << colored_input;
    
    // Clear any remaining characters from previous input
    if (last_rendered_input_.length() > current_input_.length()) {
        int chars_to_clear = last_rendered_input_.length() - current_input_.length();
        std::cout << std::string(chars_to_clear, ' ');
     }

    // Position cursor correctly within the input
    int cursor_screen_pos = esql::UTF8Processor::display_width(
        esql::UTF8Processor::strip_ansi(current_prompt)) +
        esql::UTF8Processor::display_width(
            esql::UTF8Processor::strip_ansi(current_input_.substr(0, cursor_pos_)));

    // Move cursor to correct position relative to prompt start
    std::cout << "\033[" << prompt_row_ << ";" << (prompt_col_ + cursor_screen_pos) << "H";

    std::cout.flush();

    // Update last rendered state
    last_rendered_input_ = current_input_;
    last_cursor_pos_ = cursor_pos_;
    current_prompt_ = current_prompt;
}

/*void ModernShell::update_screen() {
    // Move to where the prompt is currently located
    move_to_prompt_position();

    // Fish-like optimization: only redraw what changed
    
    std::string current_prompt = build_prompt();
    std::string colored_input = highlighter_.highlight(current_input_);

    // Clear from prompt to end of line
     std::cout << "\033[K" << current_prompt;
    
    // Calculate prompt width (without ANSI codes)
    std::string clean_prompt = esql::UTF8Processor::strip_ansi(current_prompt);
    int prompt_width = esql::UTF8Processor::display_width(clean_prompt);
    
    // Move to input area (below banner)
    //std::cout << "\033[16;1H"; // Fixed position below banner
    
    // Clear the current input line and redraw prompt
    //std::cout << "\033[K" << current_prompt;
    
    // Output colored input
    std::cout << colored_input;
    
    // Clear any remaining characters from previous input
    if (last_rendered_input_.length() > current_input_.length()) {
        int chars_to_clear = last_rendered_input_.length() - current_input_.length();
        std::cout << std::string(chars_to_clear, ' ');
    }
    
    // Position cursor correctly
    int cursor_screen_pos = prompt_width + esql::UTF8Processor::display_width(
        esql::UTF8Processor::strip_ansi(current_input_.substr(0, cursor_pos_)));
    
    // Move cursor to correct position
    std::cout << "\033[16;" << (cursor_screen_pos + 1) << "H";

    
    std::cout.flush();
    
    // Update last rendered state
    last_rendered_input_ = current_input_;
    last_cursor_pos_ = cursor_pos_;
    current_prompt_ = current_prompt;
}*/

void ModernShell::refresh_display(bool force_redraw) {
    if (force_redraw || 
        current_input_ != last_rendered_input_ || 
        cursor_pos_ != last_cursor_pos_ ||
        build_prompt() != current_prompt_) {
        update_screen();
    }
}

void ModernShell::handle_enter() {
    if (current_input_.empty()) {
        // Empty line - just show new prompt
        std::cout << "\n";
        move_to_prompt_position();
        print_prompt();
        update_prompt_position();
        refresh_display(true);
        return;
    }

    // Save command
    std::string command_to_execute = current_input_;
    
    // Move to new line
    std::cout << "\n";
    
    // Add to history
    history_.add(command_to_execute);
    
    // Execute command
    execute_command(command_to_execute);
    
    // Reset input state
    current_input_.clear();
    cursor_pos_ = 0;
    history_.reset_navigation();
    
    // Print new prompt after command output
    std::cout << "\n"; // Ensure we re on a new line
    print_prompt();
    update_prompt_position();

    // Prepare for next input
    if (!should_exit_) {
        refresh_display(true);
    }
}

void ModernShell::handle_character(char c) {
    // Insert character at cursor position
    current_input_.insert(cursor_pos_, 1, c);
    cursor_pos_++;
    refresh_display();
}

void ModernShell::handle_backspace() {
    if (cursor_pos_ == 0 || current_input_.empty()) return;
    
    // Move cursor back one character (considering UTF-8)
    cursor_pos_ = esql::UTF8Processor::prev_char_boundary(current_input_, cursor_pos_);
    
    // Delete character at cursor position
    current_input_ = esql::UTF8Processor::delete_char_at_byte(current_input_, cursor_pos_);
    
    refresh_display();
}

void ModernShell::handle_tab() {
    // Basic tab completion - insert 4 spaces
    std::string completion = "    ";
    current_input_.insert(cursor_pos_, completion);
    cursor_pos_ += completion.size();
    refresh_display();
}

void ModernShell::handle_navigation(esql::KeyCode key) {
    switch (key) {
        case esql::KeyCode::Left:
            if (cursor_pos_ > 0) {
                cursor_pos_ = esql::UTF8Processor::prev_char_boundary(current_input_, cursor_pos_);
                refresh_display();
            }
            break;
            
        case esql::KeyCode::Right:
            if (cursor_pos_ < current_input_.size()) {
                cursor_pos_ = esql::UTF8Processor::next_char_boundary(current_input_, cursor_pos_);
                refresh_display();
            }
            break;
            
        case esql::KeyCode::Up: {
            std::string history_item = history_.navigate_up();
            if (!history_item.empty()) {
                current_input_ = history_item;
                cursor_pos_ = current_input_.size();
                refresh_display();
            }
            break;
        }
            
        case esql::KeyCode::Down: {
            std::string history_item = history_.navigate_down();
            current_input_ = history_item;
            cursor_pos_ = current_input_.size();
            refresh_display();
            break;
        }
            
        case esql::KeyCode::Home:
            cursor_pos_ = 0;
            refresh_display();
            break;
            
        case esql::KeyCode::End:
            cursor_pos_ = current_input_.size();
            refresh_display();
            break;
            
        default:
            break;
    }
}

void ModernShell::reset_abandoning_line() {
    // Simple implementation - move to new line
    std::cout << "\n";
}

void ModernShell::reset_line() {
    // Clear current line and reposition cursor
    std::cout << "\r\033[K";
}

// Utility functions
esql::KeyCode ModernShell::convert_char_to_keycode(char c) {
    switch (c) {
        case '\n': case '\r': return esql::KeyCode::Enter;
        case '\t': return esql::KeyCode::Tab;
        case 127: return esql::KeyCode::Backspace;
        case 8: return esql::KeyCode::Delete;
        case 4: return esql::KeyCode::CtrlD;
        case 12: return esql::KeyCode::CtrlL;
        case 18: return esql::KeyCode::CtrlR;
        case '\033': return handle_escape_sequence();
        default:
            if (c >= 32 && c <= 126) return esql::KeyCode::Character;
            return esql::KeyCode::None;
    }
}

esql::KeyCode ModernShell::handle_escape_sequence() {
    char seq[3] = {0};
    
    // Use non-blocking read with timeout
    struct timeval tv = {0, 100000}; // 100ms timeout
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0) {
        return esql::KeyCode::Escape;
    }
    
    if (read(STDIN_FILENO, &seq[0], 1) != 1) {
        return esql::KeyCode::Escape;
    }
    
    if (seq[0] == '[') {
        if (read(STDIN_FILENO, &seq[1], 1) != 1) {
            return esql::KeyCode::Escape;
        }
        
        switch (seq[1]) {
            case 'A': return esql::KeyCode::Up;
            case 'B': return esql::KeyCode::Down;
            case 'C': return esql::KeyCode::Right;
            case 'D': return esql::KeyCode::Left;
            case 'H': return esql::KeyCode::Home;
            case 'F': return esql::KeyCode::End;
            case '1': 
                if (read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == '~') 
                    return esql::KeyCode::Home;
                break;
            case '4': 
                if (read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == '~') 
                    return esql::KeyCode::End;
                break;
        }
    }
    
    return esql::KeyCode::None;
}

bool ModernShell::handle_possible_resize() {
    static int last_width = -1, last_height = -1;
    int width, height;
    terminal_.get_terminal_size(width, height);
    
    if (width != last_width || height != last_height) {
        last_width = width;
        last_height = height;
        terminal_width_ = width;
        terminal_height_ = height;
        
        // Clear and redraw
        clear_screen();
        print_banner();
                // Reset prompt position
        prompt_row_ = 19;
        prompt_col_ = 1;
        move_to_prompt_position();
        print_prompt();
        update_prompt_position();
        refresh_display(true);
        return true;
    }
    return false;
}

void ModernShell::run_termux_fallback() {
    if (use_colors_) {
        std::cout << esql::colors::BLUE << "Termux mode activated (using line-based input)\n\n" 
                  << esql::colors::RESET;
    }
    
    while (!should_exit_) {
        print_prompt();
        
        std::string line;
        std::getline(std::cin, line);
        
        if (line.empty()) continue;
        if (line == "exit" || line == "quit") {
            break;
        }
        
        execute_command(line);
    }
}

void ModernShell::execute_command(const std::string& command) {
    if (command.empty()) return;
    
    std::string upper_cmd = command;
    std::transform(upper_cmd.begin(), upper_cmd.end(), upper_cmd.begin(), ::toupper);
    
    if (upper_cmd == "EXIT" || upper_cmd == "QUIT") {
        should_exit_ = true;
        return;
    } else if (upper_cmd == "HELP") {
        show_help();
    } else if (upper_cmd == "CLEAR") {
        clear_screen();
        print_banner();
        // Reset prompt position to below banner
        prompt_row_ = 19;
        prompt_col_ = 1;
        move_to_prompt_position();
        print_prompt();
        update_prompt_position();
        refresh_display(true);
        return;
    } else {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            auto [result, duration] = db_.executeQuery(command);
            auto end = std::chrono::high_resolution_clock::now();
            double actual_duration = std::chrono::duration<double>(end - start).count();
            
            // Handle database switching
            if (upper_cmd.find("USE ") == 0) {
                size_t use_pos = upper_cmd.find("USE ");
                if (use_pos != std::string::npos) {
                    std::string db_name = command.substr(use_pos + 4);
                    if (!db_name.empty() && db_name.back() == ';') {
                        db_name.pop_back();
                    }
                    db_name.erase(0, db_name.find_first_not_of(" \t"));
                    db_name.erase(db_name.find_last_not_of(" \t") + 1);
                    
                    if (!db_name.empty()) {
                        current_db_ = db_name;
                        highlighter_.set_current_database(current_db_);
                    }
                }
            }
            
            // OUTPUT RESULTS DIRECTLY - don't route through renderer
            print_results(result, actual_duration);
            
        } catch (const std::exception& e) {
            std::cerr << (use_colors_ ? esql::colors::RED : "") << "Error: " << e.what() 
                      << (use_colors_ ? esql::colors::RESET : "") << "\n";
        }
    }
    
    // Ensure we're ready for next input
    //std::cout << std::endl;
}

void ModernShell::print_banner() {
    terminal_.clear_screen();
    
    if (use_colors_) {
        std::cout << esql::colors::GRAY;
        std::cout << "   ███████╗███████╗ ██████╗ ██╗   \n";
        std::cout << "   ██╔════╝██╔════╝██╔═══██╗██║   \n";
        std::cout << "   █████╗  ███████╗██║   ██║██║   \n";
        std::cout << "   ██╔══╝  ╚════██║██║   ██║██║   \n";
        std::cout << "   ███████╗███████║╚██████╔╝███████╗\n";
        std::cout << "   ╚══════╝╚══════╝ ╚═════╝ ╚══════╝\n";
        std::cout << esql::colors::RESET;
        
        std::cout << esql::colors::CYAN << "╔═══════════════════════════════════════╗\n";
        std::cout << "║    " << esql::colors::MAGENTA << "E N H A N C E D   ES Q L   S H E L L" 
                  << esql::colors::CYAN << "  ║\n";
        std::cout << "║        " << esql::colors::YELLOW << "H4CK3R  STYL3  V3RSI0N" 
                  << esql::colors::CYAN << "         ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n" << esql::colors::RESET;

        std::cout << esql::colors::RED << "[*] "<< esql::colors::CYAN 
                  << "Type 'help' for commands, 'exit' to quit\n";
        std::cout << esql::colors::RED << "[*] " << esql::colors::CYAN 
                  << "Initializing ESQL Database Matrix...\n";
        std::cout << esql::colors::RED << "[*] " << esql::colors::MAGENTA 
                  << "Quantum ESQL Processor: ONLINE\n";
        std::cout << esql::colors::RED << "[*] " << esql::colors::GRAY 
                  << "Syntax Highlighting: ACTIVATED\n" << esql::colors::RESET;

        //First animation - ALWAYS on one line
        //std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::CYAN;
        std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::RESET; // Just the prefix in magenta
        std::cout.flush();
        
        ConsoleAnimator animator1(terminal_width_);
        animator1.animateText("Forged from the fires of performance and for the warriors of the digital age", 4000);
        
        // Second animation - ALWAYS on one line
        //std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::CYAN;
        std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::RESET; // Just the prefix in magenta
        std::cout.flush();
        
        WaveAnimator animator2(terminal_width_);
        animator2.waveAnimation("accessing the esql framework console", 2);

        // Connected line after animations
        std::cout << esql::colors::MAGENTA << "[+] "<< esql::colors::CYAN 
                  << "Connected to: " << (use_colors_ ? esql::colors::GRAY : "") 
                  << current_db_ << esql::colors::GREEN << "•" 
                  << (use_colors_ ? esql::colors::RESET : "") << "\n\n";
    } else {
        std::cout << "ESQL SHELL - Enhanced Query Language Shell\n";
        std::cout << "Connected to: " << current_db_ << "\n\n";
    }
}
void ModernShell::print_prompt() {
    std::cout << build_prompt();
}

void ModernShell::clear_screen() {
    terminal_.clear_screen();
}

std::string ModernShell::build_prompt() const {
    std::string time_str = get_current_time();
    std::string prompt;
    
    if (use_colors_) {
        prompt = std::string(esql::colors::YELLOW) + "[" + time_str + "] " + 
                 esql::colors::RESET +
                 std::string(esql::colors::GREEN) + "• " + 
                 esql::colors::RESET +
                 std::string(esql::colors::GRAY) + current_db_ + 
                 esql::colors::RESET + "> ";
    } else {
        prompt = "[" + time_str + "] • " + current_db_ + "> ";
    }
    
    return prompt;
}

std::string ModernShell::get_current_time() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M");
    return ss.str();
}

void ModernShell::show_help() {
    std::cout << "\n" << (use_colors_ ? esql::colors::CYAN : "") << "Available commands:" 
              << (use_colors_ ? esql::colors::RESET : "") << "\n";
    
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
        std::cout << "  " << (use_colors_ ? esql::colors::MAGENTA : "") << cmd 
                  << (use_colors_ ? esql::colors::GREEN : "") << " - " << desc 
                  << (use_colors_ ? esql::colors::RESET : "") << "\n";
    }
    std::cout << "\n";
}

void ModernShell::set_current_database(const std::string& db_name) {
    current_db_ = db_name;
    highlighter_.set_current_database(db_name);
}

void ModernShell::print_results(const ExecutionEngine::ResultSet& result, double duration) {
    if (result.columns.empty()) {
        std::cout << (use_colors_ ? esql::colors::GREEN : "") << "Query executed successfully.\n" << (use_colors_ ? esql::colors::RESET : "");
        return;
    }
    if (result.columns.size() == 2 && (result.columns[0] == "Property" || result.columns[0] == "Database Structure")) {
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

    std::cout << (use_colors_ ? esql::colors::CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n";

    std::cout << "|";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << (use_colors_ ? esql::colors::MAGENTA : "") << std::left << std::setw(widths[i]) << result.columns[i] << (use_colors_ ? esql::colors::CYAN : "") << "|";
    }
    
    std::cout << (use_colors_ ? esql::colors::RESET : "") << "\n";

    std::cout << (use_colors_ ? esql::colors::CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n" << (use_colors_ ? esql::colors::RESET : "");

    for (const auto& row : result.rows) {
        std::cout << (use_colors_ ? esql::colors::CYAN : "") << "|" << (use_colors_ ? esql::colors::RESET : "");
        for (size_t i = 0; i < row.size(); ++i) {
            std::string display_value = row[i];
            if (display_value.length() > widths[i] - 2) {
                display_value = display_value.substr(0, widths[i] - 5) + "...";
            }
            std::cout << (use_colors_ ? esql::colors::YELLOW : "") << std::left << std::setw(widths[i]) << display_value << (use_colors_ ? esql::colors::CYAN : "") << "|" << (use_colors_ ? esql::colors::RESET : "");
        }
        std::cout << "\n";
    }

    std::cout << (use_colors_ ? esql::colors::CYAN : "") << "+";
    for (size_t i = 0; i < result.columns.size(); ++i) {
        std::cout << std::string(widths[i], '-');
        if (i < result.columns.size() - 1) std::cout << "+";
    }
    std::cout << "+\n" << (use_colors_ ? esql::colors::RESET : "");

    std::cout << (use_colors_ ? esql::colors::GRAY : "") << "> " << result.rows.size() << " row" << (result.rows.size() != 1 ? "s" : "") << " in " << std::fixed << std::setprecision(4) << duration << "s" << (use_colors_ ? esql::colors::RESET : "") << "\n";
}

void ModernShell::print_structure_results(const ExecutionEngine::ResultSet& result, double duration) {
    std::cout << (use_colors_ ? esql::colors::CYAN : "") << "==============================================================\n";
    std::cout << (use_colors_ ? esql::colors::MAGENTA : "") << "DATABASE STRUCTURE REPORT\n";
    std::cout << (use_colors_ ? esql::colors::CYAN : "") << "==============================================================\n" << (use_colors_ ? esql::colors::RESET : "");

    std::string current_section = "";

    for (const auto& row : result.rows) {
        if (row.size() < 2) continue;
        const std::string& left = row[0];
        const std::string& right = row[1];
        if (right.empty() && (left.find("COLUMNS") != std::string::npos ||
                    left.find("CONSTRAINTS") != std::string::npos ||
                    left.find("TABLE DETAILS") != std::string::npos ||
                    left.find("STORAGE INFO") != std::string::npos)) {
            std::cout << "\n" << (use_colors_ ? esql::colors::GREEN : "") << "=== " << left << " " << std::string(50 - left.length(), '=') << "\n" << (use_colors_ ? esql::colors::RESET : "");
            current_section = left;
            continue;
        }
        if (left == "---" && right == "---") {
            std::cout << (use_colors_ ? esql::colors::GRAY : "") << std::string(60, '-') << "\n" << (use_colors_ ? esql::colors::RESET : "");
            continue;
        }
        if (current_section.find("COLUMNS") != std::string::npos && left.empty() && right.find("Name | Type") != std::string::npos) {
            std::cout << (use_colors_ ? esql::colors::CYAN : "") << std::left << std::setw(56) << right << (use_colors_ ? esql::colors::RESET : "") << "\n";
            std::cout << std::string(56, '-') << "\n" << (use_colors_ ? esql::colors::RESET : "");
            continue;
        }
        if (!left.empty() && !right.empty()) {
            if (left == right) {
                std::cout << (use_colors_ ? esql::colors::MAGENTA : "") << std::left << std::setw(56) << left << (use_colors_ ? esql::colors::RESET : "") << "\n";
            } else {
                std::cout << (use_colors_ ? esql::colors::YELLOW : "") << std::left << std::setw(20) << left << (use_colors_ ? esql::colors::CYAN : "") << " : " << (use_colors_ ? esql::colors::GRAY : "") << std::left << std::setw(33) << right << (use_colors_ ? esql::colors::RESET : "") << "\n";
            }
        } else if (left.empty() && !right.empty()) {
            std::cout << "   " << (use_colors_ ? esql::colors::BLUE : "") << std::left << std::setw(53) << right << (use_colors_ ? esql::colors::RESET : "") << "\n";
        }
    }
    
    std::cout << (use_colors_ ? esql::colors::CYAN : "") << std::string(60, '=') << "\n" << (use_colors_ ? esql::colors::RESET : "");
    std::cout << (use_colors_ ? esql::colors::GRAY : "") << "> Report generated in " << std::fixed << std::setprecision(4) << duration << "s" << (use_colors_ ? esql::colors::RESET : "") << "\n";
}
