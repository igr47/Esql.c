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
    : db_(db), current_db_("default"), completion_engine_(db), autosuggestion_manager_(history_)  {
    
    // Initialize terminal
    terminal_.enable_raw_mode();
    terminal_.get_terminal_size(terminal_width_, terminal_height_);
    
    highlighter_.enable_colors(use_colors_);
    highlighter_.set_current_database(current_db_);
    completion_engine_.set_current_database(current_db_);

    // Initial prompt positions is below the banner
    prompt_row_ = 33;
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
    update_prompt_position();
    
    refresh_display(true);

    // Buffer for detecting escape sequence responses
    std::string escape_buffer;
    bool in_escape_sequence = false;

    while (!should_exit_) {
        char c;
        ssize_t n = read(STDIN_FILENO, &c, 1);

        if (n != 1) continue;

                // Handle escape sequence responses from cursor position queries
        if (in_escape_sequence) {
            escape_buffer += c;

            // Check if we have a complete cursor position response: \033[<row>;<col>R
            if (c == 'R' && escape_buffer.length() >= 6) {
                // This is a cursor position response, ignore it
                in_escape_sequence = false;
                escape_buffer.clear();
                continue;
            }

                       // If buffer gets too long, assume it's not a cursor position response
            if (escape_buffer.length() > 15) {
                in_escape_sequence = false;
                escape_buffer.clear();
            } else {
                continue; // Still processing escape sequence
            }
        }
        // Handle escape sequences (arrow keys)
        if (c == '\033') {
            // Read the next characters with timeout
            struct timeval tv = {0, 50000}; // 50ms timeout
            fd_set fds;
            
            // Check if there are more characters available
            FD_ZERO(&fds);
            FD_SET(STDIN_FILENO, &fds);
            
            if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
                char seq[2];
                // Read the '[' character
                if (read(STDIN_FILENO, &seq[0], 1) == 1 && seq[0] == '[') {
                    // Read the direction character
                    if (read(STDIN_FILENO, &seq[1], 1) == 1) {
                        // Process the arrow key
                        switch (seq[1]) {
                            case 'A': // Up arrow
                                handle_navigation(esql::KeyCode::Up);
                                continue; // Skip the rest of the loop
                            case 'B': // Down arrow
                                handle_navigation(esql::KeyCode::Down);
                                continue;
                            case 'C': // Right arrow
                                handle_navigation(esql::KeyCode::Right);
                                continue;
                            case 'D': // Left arrow
                                handle_navigation(esql::KeyCode::Left);
                                continue;
                            case 'H': // Home
                                handle_navigation(esql::KeyCode::Home);
                                continue;
                            case 'F': // End
                                handle_navigation(esql::KeyCode::End);
                                continue;
                        }
                    }
                }
            }
            
            // If we get here, it's just the Escape key
            if (in_multiline_mode_) {
                in_multiline_mode_ = false;
                multiline_buffer_.clear();
                current_input_.clear();
                cursor_pos_ = 0;
                std::cout << "\n";
                print_prompt();
                update_prompt_position();
                refresh_display(true);
            }
            continue;
        }

        // Handle regular characters
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
            case esql::KeyCode::CtrlD:
                should_exit_ = true;
                break;
            case esql::KeyCode::CtrlL:
                clear_screen();
                print_banner();
                refresh_display(true);
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

void ModernShell::ensure_input_space() {
    // Get current cursor position after command output
    int current_row, current_col;
    terminal_.get_cursor_position(current_row, current_col);

    // Calculate how much space we need for the next input
    // We need at least 2-3 lines for input (prompt + possible wrapped lines)
    int needed_space = 3;
    int available_space = terminal_height_ - current_row;

    if (available_space < needed_space) {
        scroll_input_area(needed_space - available_space);
    }
}

void ModernShell::scroll_input_area(int lines_to_scroll) {
    if (lines_to_scroll <= 0) return;

    // Use the proper ANSI escape sequence to scroll the screen
    // This is more efficient than multiple single-line scrolls
    std::cout << "\033[" << lines_to_scroll << "S";

    // Adjust prompt position to account for scrolling
    prompt_row_ = std::max(1, prompt_row_ - lines_to_scroll);

    std::cout.flush();
}


/*
 * Methods to ensure that when auto suggetion  spans multiple lines the lines beyond the current input are clered well so as to avoid displaying them even when the suggestion becomes small
*/
int ModernShell::calculate_rendered_lines(const std::string& input, const std::string& prompt) {
    if (input.empty()) return 1;

    // Calculate prompt width without ANSI codes
    std::string clean_prompt = esql::UTF8Processor::strip_ansi(prompt);
    int prompt_width = esql::UTF8Processor::display_width(clean_prompt);

    // Calculate total display width including autosuggestion
    std::string display_text = input;
    if (current_suggestion_.active && input == current_suggestion_.prefix) {
        display_text += current_suggestion_.suggestion.substr(current_suggestion_.display_start);
    }

    int display_width = esql::UTF8Processor::display_width(display_text);

    // First line available width (accounting for prompt)
    int first_line_width = terminal_width_ - prompt_width;
    if (first_line_width < 1) first_line_width = 1;

    // Continuation line width (full terminal width for wrapped content)
    int continuation_width = terminal_width_;

    // Calculate total lines needed
    int lines = 1;
    int remaining_width = display_width;

    if (remaining_width > first_line_width) {
        remaining_width -= first_line_width;
        lines++;

        // Calculate additional continuation lines
        while (remaining_width > continuation_width) {
            remaining_width -= continuation_width;
            lines++;
        }
    }

    return lines;
}

void ModernShell::clear_previous_lines(int previous_lines, int current_lines) {
    // If we need to clear more lines than we are currently using
    if (previous_lines > current_lines) {
        // Move to first line that needs to be cleared
        for (int i = current_lines; i < previous_lines; i++) {
                // Move to continuation line
                std::cout << "\033[" << (prompt_row_ + i) << ";1H";
                // Clear the entire line
                std::cout << "\033[K";
        }

        // Move back to the prompt position
        move_to_prompt_position();
    }
}

void ModernShell::update_screen() {
        // If in multiline mode, we need special handling for display
    if (in_multiline_mode_ && !multiline_buffer_.empty()) {
        // For multiline mode, we need to display all accumulated lines
        move_to_prompt_position();
        
        std::string current_prompt = build_prompt();
        std::cout << "\033[K" << current_prompt;
        
        // Display all previous multiline buffers
        for (size_t i = 0; i < multiline_buffer_.size(); ++i) {
            std::string colored_line = highlighter_.highlight(multiline_buffer_[i]);
            std::cout << colored_line << "\n";

                        // Print continuation prefix
            if (use_colors_) {
                std::cout << esql::colors::GRAY << " -> " << esql::colors::RESET;
            } else {
                std::cout << " -> ";
            }
        }

        // Display current input with highlighting
        //istd::string colored_input = highlighter_.highlight(current_input_);
        std::string colored_input = render_with_suggestion(current_input_, current_suggestion_);
        std::cout << colored_input;

        // Clear any remaining characters from previous input
        if (last_rendered_input_.length() > current_input_.length()) {
            int chars_to_clear = last_rendered_input_.length() - current_input_.length();
            std::cout << std::string(chars_to_clear, ' ');
        }

        // Position cursor correctly
        int cursor_screen_pos = esql::UTF8Processor::display_width(
            esql::UTF8Processor::strip_ansi(" -> ")) +  // Continuation prefix
            esql::UTF8Processor::display_width(
                esql::UTF8Processor::strip_ansi(current_input_.substr(0, cursor_pos_)));

        // Calculate which row we're on (prompt row + number of multiline buffers)
        int current_row = prompt_row_ + static_cast<int>(multiline_buffer_.size());
        std::cout << "\033[" << current_row << ";" << (prompt_col_ + cursor_screen_pos) << "H";

        std::cout.flush();

        // Update last rendered state
        last_rendered_input_ = current_input_;
        last_cursor_pos_ = cursor_pos_;
        current_prompt_ = current_prompt;
        return;
    }
    // Move to where the prompt is currently located
    move_to_prompt_position();
    
    std::string current_prompt = build_prompt();
    //std::string colored_input = highlighter_.highlight(current_input_);
    std::string colored_input = render_with_suggestion(current_input_, current_suggestion_);

    // Calculate how many lines the previous render occupied
    int previous_lines = calculate_rendered_lines(last_rendered_input_, current_prompt_);
    int current_lines = calculate_rendered_lines(current_input_, current_prompt);

       // Clear ALL lines that might have been used previously
    for (int i = 0; i < previous_lines; i++) {
        if (i > 0) {
            // Move to continuation line
            std::cout << "\033[" << (prompt_row_ + i) << ";1H";
        }
        // Clear the entire line
        std::cout << "\033[K";
    }

        // Move back to prompt position
    move_to_prompt_position();
    
    // Clear from prompt position to end of line
    std::cout << "\033[K" << current_prompt;
    
    // Output colored input
    std::cout << colored_input;
    
    // Clear any remaining characters from previous input
    size_t current_total_length = current_input_.length();
    if (current_suggestion_.active && current_input_ == current_suggestion_.prefix) {
        current_total_length += current_suggestion_.suggestion.length() - current_suggestion_.display_start;
    }

    size_t last_total_length = last_rendered_input_.length();
    if (current_suggestion_.active && last_rendered_input_ == current_suggestion_.prefix) {
        last_total_length += current_suggestion_.suggestion.length() - current_suggestion_.display_start;
    }
    
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



void ModernShell::refresh_display(bool force_redraw) {
    ensure_input_space();
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

    // Check if we should continue in multiline mode
    // We continue if current line doesn't end with semicolon
    // We execute if current line ends with semicolon
    bool should_continue_multiline = current_input_.back() != ';';

    if (should_continue_multiline) {
        // Enter multiline mode or continue in multiline mode
        if (!in_multiline_mode_) {
            // Starting multiline mode - save first line
            multiline_buffer_.clear();
            in_multiline_mode_ = true;
        }

        multiline_buffer_.push_back(current_input_);

        // Move to new line with continuation prefix
        std::cout << "\n";

        // Print continuation prompt
        if (use_colors_) {
            std::cout << esql::colors::GRAY << " -> " << esql::colors::RESET;
        } else {
            std::cout << " -> ";
        }

        // Reset input state for next line
        current_input_.clear();
        cursor_pos_ = 0;

        // Update display for the new continuation line
        refresh_display(true);
        return;
    }

    // We have a complete command (ends with semicolon and we're ready to execute)
    std::string command_to_execute;
    std::string history_command;

    if (in_multiline_mode_) {
        // Combine all multiline buffers with the current line
        for (const auto& line : multiline_buffer_) {
            command_to_execute += line + " ";
        }
        command_to_execute += current_input_;
        history_command = command_to_execute;

        // Reset multiline state
        in_multiline_mode_ = false;
        multiline_buffer_.clear();
    } else {
        // Single line command
        command_to_execute = current_input_;
        history_command = current_input_;
    }

    // Remove trailing semicolon and trim
    if (!command_to_execute.empty() && command_to_execute.back() == ';') {
        command_to_execute.pop_back();
    }
    // Trim whitespace
    command_to_execute.erase(0, command_to_execute.find_first_not_of(" \t"));
    command_to_execute.erase(command_to_execute.find_last_not_of(" \t") + 1);

    // Move to new line
    std::cout << "\n";

    // Add to history
    history_.add(history_command);

    // Execute command
    execute_command(command_to_execute);

     ensure_input_space();

    // Reset input state
    current_input_.clear();
    cursor_pos_ = 0;
    history_.reset_navigation();

    // Print new prompt after command output
    //std::cout << "\n"; // Ensure we're on a new line
    int current_row, current_col;
    terminal_.get_cursor_position(current_row, current_col);
    if (current_row >= terminal_height_ - 2) {
        std::cout << "\n";
    } else {
        std::cout << "\r\n"; // Just carriage return + newline, no extra spaces
    }
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

    // Updte auto suggestion
    update_autosuggestion();

    refresh_display();
}

void ModernShell::handle_backspace() {
    if (cursor_pos_ == 0 || current_input_.empty()) return;
    
    // Move cursor back one character (considering UTF-8)
    cursor_pos_ = esql::UTF8Processor::prev_char_boundary(current_input_, cursor_pos_);
    
    // Delete character at cursor position
    current_input_ = esql::UTF8Processor::delete_char_at_byte(current_input_, cursor_pos_);

    // Update autosuggestion
    update_autosuggestion();
    
    refresh_display();
}

void ModernShell::handle_tab() {
    // Get completions for current input and cursor position
    auto completions = completion_engine_.get_completions(current_input_, cursor_pos_);

    if (completions.empty()) {
        // No completions found, insert spaces as fall back
        std::string completions = " ";
        current_input_.insert(cursor_pos_, completions);
        cursor_pos_ += completions.size();
        refresh_display();
    } else if (completions.size() == 1) {
        // Single completion - insert it
        const std::string& completion = completions[0];

        // Calculate the part to insert (after the current prefix)
        size_t prefix_length = get_current_word_prefix().length();
        std::string to_insert = completion.substr(prefix_length);

        current_input_.insert(cursor_pos_, to_insert);
        cursor_pos_ += to_insert.length();
        refresh_display();
    } else {
        // Multiple completions - show the
        show_completions(completions);
    }
}

// Helper to get the word prefix
std::string ModernShell::get_current_word_prefix() {
    if (cursor_pos_ == 0 || current_input_.empty()) return "";
    
    size_t word_start = cursor_pos_;
    while (word_start > 0 && 
           (std::isalnum(current_input_[word_start-1]) || 
            current_input_[word_start-1] == '_' || 
            current_input_[word_start-1] == '.')) {
        --word_start;
    }
    
    return current_input_.substr(word_start, cursor_pos_ - word_start);
}

// Method to show completion list
void ModernShell::show_completions(const std::vector<std::string>& completions) {
    if (completions.empty()) return;
    
    std::cout << "\n"; // Move to new line
    
    // Calculate display parameters
    int max_width = 0;
    for (const auto& comp : completions) {
        int width = esql::UTF8Processor::display_width(comp);
        if (width > max_width) max_width = width;
    }
    
    max_width += 2; // Add padding

    int items_per_row = std::max(1, terminal_width_ / max_width);
    int row_count = (completions.size() + items_per_row - 1) / items_per_row;

    // Display completions in columns
    for (int row = 0; row < row_count; ++row) {
        for (int col = 0; col < items_per_row; ++col) {
            size_t index = row + col * row_count;
            if (index < completions.size()) {
                std::cout << std::left << std::setw(max_width) << completions[index];
            }
        }
        std::cout << "\n";
            }

    // Redisplay prompt and current input
    print_prompt();
    std::cout << current_input_;

    // Reposition cursor
    int cursor_screen_pos = esql::UTF8Processor::display_width(
        esql::UTF8Processor::strip_ansi(build_prompt())) +
        esql::UTF8Processor::display_width(
            esql::UTF8Processor::strip_ansi(current_input_.substr(0, cursor_pos_)));

    std::cout << "\033[" << prompt_row_ << ";" << (prompt_col_ + cursor_screen_pos) << "H";
    std::cout.flush();
}

void ModernShell::update_autosuggestion() {
    current_suggestion_ = autosuggestion_manager_.get_suggestion(current_input_);
}

void ModernShell::accept_autosuggestion() {
    if (current_suggestion_.active) {
        current_input_ = autosuggestion_manager_.accept_suggestion(current_input_, current_suggestion_);
        cursor_pos_ = current_input_.size();
        clear_autosuggestion();
        refresh_display();
    }
}

void ModernShell::clear_autosuggestion() {
    current_suggestion_.active = false;
    current_suggestion_.suggestion.clear();
    current_suggestion_.prefix.clear();
}

std::string ModernShell::render_with_suggestion(const std::string& input,
                                              const esql::AutoSuggestion& suggestion) {
    if (!suggestion.active || input != suggestion.prefix) {
        return highlighter_.highlight(input);
    }

    // Render input normally, then add suggestion in gray
    std::string result = highlighter_.highlight(input);

    if (suggestion.display_start < suggestion.suggestion.length()) {
        std::string suggestion_part = suggestion.suggestion.substr(suggestion.display_start);
        result += esql::colors::GRAY + suggestion_part + esql::colors::RESET;
    }

    return result;
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
            } else if (cursor_pos_ == current_input_.size() && current_suggestion_.active) {
                accept_autosuggestion();
            }
            break;
            
        case esql::KeyCode::Up: {
            std::string history_item = history_.navigate_up();
            if (!history_item.empty()) {
                current_input_ = history_item;
                cursor_pos_ = current_input_.size();

                // Clear auto suggestion when using history navigation
                clear_autosuggestion();
                refresh_display();
            }
            break;
        }
            
        case esql::KeyCode::Down: {
            std::string history_item = history_.navigate_down();
            current_input_ = history_item;
            cursor_pos_ = current_input_.size();
            // Clear autosuggestion when using history navigation
            clear_autosuggestion();
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
        prompt_row_ = 33;
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
        prompt_row_ = 33;
        prompt_col_ = 1;
        move_to_prompt_position();
        print_prompt();
        update_prompt_position();

        // Reset multiline state
        in_multiline_mode_ = false;
        multiline_buffer_.clear();

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
                        completion_engine_.set_current_database(current_db_); 
                    }
                }
            }

            if (upper_cmd.find("CREATE ") == 0 || upper_cmd.find("DROP ") == 0 || upper_cmd.find("ALTER ") == 0) {
                completion_engine_.refresh_metadata();
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
    ensure_input_space();
}

void ModernShell::print_banner() {
    terminal_.clear_screen();

    if (use_colors_) {
        // Position cursor at top
        std::cout << "\033[1;1H";

        // Create Phoenix animator
        PhoenixAnimator phoenix_animator(terminal_width_);

        // ESQL logo (6 lines)
        std::vector<std::string> esql_logo = {
            "   ███████╗███████╗ ██████╗ ██╗   ",
            "   ██╔════╝██╔════╝██╔═══██╗██║   ",
            "   █████╗  ███████╗██║   ██║██║   ",
            "   ██╔══╝  ╚════██║██║   ██║██║   ",
            "   ███████╗███████║╚██████╔╝███████╗",
            "   ╚══════╝╚══════╝ ╚═════╝ ╚══════╝"
        };

        int esql_height = esql_logo.size();
        int phoenix_height = phoenix_animator.get_current_frame().size();

        // Calculate positioning - align bases with 0 free space
        int phoenix_start_row = 1; // Phoenix starts at row 1
        int esql_start_row = 1 + (phoenix_height - esql_height); // ESQL starts at row 14 (1 + 13)

        int phoenix_start_col = terminal_width_ - 45;
        if (phoenix_start_col < 40) phoenix_start_col = 40;

        // Draw Phoenix art (19 lines starting from row 1)
        for (int i = 0; i < phoenix_height; ++i) {
            std::cout << "\033[" << (phoenix_start_row + i) << ";" << phoenix_start_col << "H";
            std::string colored_line = phoenix_animator.apply_gradient(
                phoenix_animator.get_current_frame()[i], i * 2);
            std::cout << colored_line;
        }

        // Draw ESQL logo (6 lines starting from row 14)
        for (int i = 0; i < esql_height; ++i) {
            std::cout << "\033[" << (esql_start_row + i) << ";1H";
            std::cout << esql::colors::GRAY << esql_logo[i] << esql::colors::RESET;
        }

        // Header box starts below the combined art (row 20)
        int header_start_line = phoenix_start_row + phoenix_height + 1; // 1 + 19 + 1 = 21
        std::cout << "\033[" << header_start_line << ";1H";

        std::cout << esql::colors::CYAN << "╔═══════════════════════════════════════╗\n";
        std::cout << "║    " << esql::colors::MAGENTA << "E N H A N C E D   ES Q L   S H E L L"
                  << esql::colors::CYAN << "  ║\n";
        std::cout << "║        " << esql::colors::YELLOW << "H4CK3R  STYL3  V3RSI0N"
                  << esql::colors::CYAN << "         ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n" << esql::colors::RESET;

        // Status messages (4 lines starting from row 25)
        int status_start_line = header_start_line + 4; // 21 + 4 = 25
        std::cout << "\033[" << status_start_line << ";1H";

        std::cout << esql::colors::RED << "[*] "<< esql::colors::CYAN
                  << "Type 'help' for commands, 'exit' to quit\n";
        std::cout << esql::colors::RED << "[*] " << esql::colors::CYAN
                  << "Initializing ESQL Database Matrix...\n";
        std::cout << esql::colors::RED << "[*] " << esql::colors::MAGENTA
                  << "Quantum ESQL Processor: ONLINE\n";
        std::cout << esql::colors::RED << "[*] " << esql::colors::GRAY
                  << "Syntax Highlighting: ACTIVATED" << esql::colors::RESET;
        std::cout.flush();

        // STEP 1: Phoenix fire effect comes to life for 3 seconds after "Syntax Highlighting"
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Small pause
        std::thread phoenix_thread([&phoenix_animator, phoenix_start_row, phoenix_start_col]() {
            phoenix_animator.animate_fire_effect(3000); // 3 seconds of fire
        });
        phoenix_thread.join(); // Wait for Phoenix animation to complete

        // STEP 2: First line animation
        int anim1_line = status_start_line + 4; // 25 + 2 = 27 (right below status messages)
        std::cout << "\033[" << anim1_line << ";1H";
        std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::RESET;
        std::cout.flush();

        ConsoleAnimator animator1(terminal_width_);
        animator1.animateText("Forged from the fires of performance for the warriors of the digital age", 4000);

        // STEP 3: Second line animation
        int anim2_line = anim1_line + 1; // 27 + 1 = 28
        std::cout << "\033[" << anim2_line << ";1H";
        std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::RESET;
        std::cout.flush();

        WaveAnimator animator2(terminal_width_);
        animator2.waveAnimation("accessing the esql framework console", 2);

        // STEP 4: Final connection line
        int conn_line = anim2_line + 1; // 28 + 1 = 29
        std::cout << "\033[" << conn_line << ";1H";
        std::cout << esql::colors::MAGENTA << "[+] "<< esql::colors::CYAN
                  << "Connected to: " << (use_colors_ ? esql::colors::GRAY : "")
                  << current_db_ << esql::colors::GREEN << "•"
                  << (use_colors_ ? esql::colors::RESET : "") << "\n\n";

        // Set prompt position below everything (row 31)
        //prompt_row_ = conn_line + 2; // 29 + 2 = 31
        //prompt_col_ = 1;

    } else {
        // Fallback for no colors
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
    completion_engine_.set_current_database(db_name);
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
