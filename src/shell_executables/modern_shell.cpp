#include "shell_includes/modern_shell.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <regex>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/select.h>
#endif

ModernShell::ModernShell(Database& db) 
    : db_(db), current_db_("default"), completion_engine_(db), autosuggestion_manager_(history_), gradient_system_()  {
    
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

    // Only include suggestion if it's active and matches current input
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
            std::cout << colored_line << "\n\033[K";

            // Print continuation prefix for the next buffer line
            if (i < multiline_buffer_.size() - 1) {
                if (use_colors_) {
                    std::cout << esql::colors::GRAY << " -> " << esql::colors::RESET;
                } else {
                    std::cout << " -> ";
                }
            }
        }

        // Print continuation prefix for current input line
        if (use_colors_) {
            std::cout << esql::colors::GRAY << " -> " << esql::colors::RESET;
        } else {
            std::cout << " -> ";
        }

        // Display current input with highlighting - no suggestions in multiline mode
        std::string colored_input = highlighter_.highlight(current_input_);
        std::cout << colored_input << "\033[K";

        // Clear any remaining characters from previous input
        if (last_rendered_input_.length() > current_input_.length()) {
            int chars_to_clear = last_rendered_input_.length() - current_input_.length();
            std::cout << std::string(chars_to_clear, ' ') << "\033[K";
        }

        // Position cursor correctly
        int cursor_screen_pos = esql::UTF8Processor::display_width(
            esql::UTF8Processor::strip_ansi(" -> ")) +
            esql::UTF8Processor::display_width(
                esql::UTF8Processor::strip_ansi(current_input_.substr(0, cursor_pos_)));

        // Calculate which row we're on (prompt row + number of multiline buffers)
        int current_row = prompt_row_ + static_cast<int>(multiline_buffer_.size());
        std::cout << "\033[" << current_row << ";" << (prompt_col_ + cursor_screen_pos) << "H";

        std::cout.flush();

        // Update tracked render lines for multiline mode
        previous_render_lines_ = static_cast<int>(multiline_buffer_.size()) + 1;

        // Update last rendered state
        last_rendered_input_ = current_input_;
        last_cursor_pos_ = cursor_pos_;
        current_prompt_ = current_prompt;
        return;
    }

    // NORMAL MODE
    move_to_prompt_position();

    std::string current_prompt = build_prompt();
    std::string colored_input = render_with_suggestion(current_input_, current_suggestion_);

    // Calculate how many lines the CURRENT render will use
    int current_lines = calculate_rendered_lines(current_input_, current_prompt);

    // Use tracked previous render lines instead of recalculating
    int previous_lines = previous_render_lines_;

    // Clear ALL previous lines, even if current uses fewer
    for (int i = 0; i < previous_lines; i++) {
        if (i > 0) {
            // Move to continuation line
            std::cout << "\033[" << (prompt_row_ + i) << ";1H";
        }
        // Clear the entire line
        std::cout << "\033[K";
    }

    // If we cleared more lines than we're using now, clear any extra lines
    if (previous_lines > current_lines) {
        for (int i = current_lines; i < previous_lines; i++) {
            std::cout << "\033[" << (prompt_row_ + i) << ";1H";
            std::cout << "\033[K";
        }
    }

    // Move back to prompt position
    move_to_prompt_position();

    // Clear from prompt position to end of line
    std::cout << "\033[K" << current_prompt;

    // Output colored input
    std::cout << colored_input << "\033[K";

    // Clear any remaining characters from previous input
    int current_display_length = esql::UTF8Processor::display_width(current_input_);
    if (current_suggestion_.active && current_input_ == current_suggestion_.prefix) {
        current_display_length += esql::UTF8Processor::display_width(
            current_suggestion_.suggestion.substr(current_suggestion_.display_start));
    }

    int last_display_length = esql::UTF8Processor::display_width(last_rendered_input_);
    if (current_suggestion_.active && last_rendered_input_ == current_suggestion_.prefix) {
        last_display_length += esql::UTF8Processor::display_width(
            current_suggestion_.suggestion.substr(current_suggestion_.display_start));
    }

    if (last_display_length > current_display_length) {
        int chars_to_clear = last_display_length - current_display_length;
        std::cout << std::string(chars_to_clear, ' ') << "\033[K";
    }

    // Position cursor correctly within the input
    int cursor_screen_pos = esql::UTF8Processor::display_width(
        esql::UTF8Processor::strip_ansi(current_prompt)) +
        esql::UTF8Processor::display_width(
            esql::UTF8Processor::strip_ansi(current_input_.substr(0, cursor_pos_)));

    // Move cursor to correct position relative to prompt start
    std::cout << "\033[" << prompt_row_ << ";" << (prompt_col_ + cursor_screen_pos) << "H";

    std::cout.flush();

    // Update the tracked previous render lines
    previous_render_lines_ = current_lines;

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
    if (current_row >= terminal_height_ - 1) {
        std::cout << "\n";
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


std::string ModernShell::render_with_spell_check(const std::string& input) {
    if (!use_colors_ || input.empty()) {
        return highlighter_.highlight(input);
    }

    auto misspellings = spell_checker_.check_spelling(input);
    if (misspellings.empty()) {
        return highlighter_.highlight(input);
    }

    // Build the result by processing the input character by character
    std::string result;
    size_t current_pos = 0;

    // Sort misspellings by start position to process in order
    std::vector<std::pair<size_t, size_t>> sorted_misspellings(misspellings.begin(), misspellings.end());
    std::sort(sorted_misspellings.begin(), sorted_misspellings.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (const auto& [start, end] : sorted_misspellings) {
        // Highlight text before this misspelling
        if (current_pos < start) {
            std::string before_misspelling = input.substr(current_pos, start - current_pos);
            result += highlighter_.highlight(before_misspelling);
        }

        // Highlight the misspelled word in red
        std::string misspelled_word = input.substr(start, end - start);
        result += esql::colors::BRIGHT_RED + misspelled_word + esql::colors::RESET;

        current_pos = end;
    }

    // Highlight any remaining text after the last misspelling
    if (current_pos < input.length()) {
        std::string remaining = input.substr(current_pos);
        result += highlighter_.highlight(remaining);
    }

    return result;
}

std::string ModernShell::render_with_suggestion(const std::string& input,
                                              const esql::AutoSuggestion& suggestion) {
    std::string base_render;

    if (!suggestion.active || input != suggestion.prefix) {
        base_render = render_with_spell_check(input);
    } else {
        // Render input with spell check, then add suggestion
        base_render = render_with_spell_check(input);

        if (suggestion.display_start < suggestion.suggestion.length()) {
            std::string suggestion_part = suggestion.suggestion.substr(suggestion.display_start);
            base_render += esql::colors::GRAY + suggestion_part + esql::colors::RESET;
        }
    }

    return base_render;
}

/*std::string ModernShell::render_with_suggestion(const std::string& input,
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
}*/

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
    //ensure_input_space();
}

ModernShell::GradientSystem::GradientSystem() {
    presets[GradientType::BLUE_OCEAN] = {
        {esql::colors::GRADIENT_BLUE_1, esql::colors::GRADIENT_BLUE_2, esql::colors::GRADIENT_BLUE_3, esql::colors::GRADIENT_BLUE_4},
        "Blue Ocean"
    };

    presets[GradientType::PURPLE_DAWN] = {
        {esql::colors::GRADIENT_PURPLE_1, esql::colors::GRADIENT_PURPLE_2, esql::colors::GRADIENT_PURPLE_3, esql::colors::GRADIENT_PURPLE_4},
        "Purple Dawn"
    };
    presets[GradientType::CYAN_AURORA] = {
        {esql::colors::GRADIENT_CYAN_1, esql::colors::GRADIENT_CYAN_2, esql::colors::GRADIENT_CYAN_3, esql::colors::GRADIENT_CYAN_4},
        "Cyan Aurora"
    };

    presets[GradientType::GREEN_FOREST] = {
        {esql::colors::GRADIENT_GREEN_1, esql::colors::GRADIENT_GREEN_2, esql::colors::GRADIENT_GREEN_3, esql::colors::GRADIENT_GREEN_4},
        "Green Forest"
    };

    presets[GradientType::ORANGE_SUNSET] = {
        {esql::colors::GRADIENT_ORANGE_1, esql::colors::GRADIENT_ORANGE_2, esql::colors::GRADIENT_ORANGE_3, esql::colors::GRADIENT_ORANGE_4},
        "Orange Sunset"
    };

    presets[GradientType::MAGENTA_NEBULA] = {
        {esql::colors::GRADIENT_MAGENTA_1, esql::colors::GRADIENT_MAGENTA_2, esql::colors::GRADIENT_MAGENTA_3, esql::colors::GRADIENT_MAGENTA_4},
        "Magenta Nebula"
    };

    presets[GradientType::PREMIUM_GOLD] = {
        {esql::colors::GRADIENT_ORANGE_1, esql::colors::PREMIUM_GOLD, esql::colors::BRIGHT_YELLOW, esql::colors::PREMIUM_GOLD},
        "Premium Gold"
    };
}

const ModernShell::GradientSystem::GradientPreset& ModernShell::GradientSystem::get_preset(GradientType type) const {
    auto it = presets.find(type);
    if (it != presets.end()) {
        return it->second;
    }
    // Fallback to blue ocean
    return presets.at(GradientType::BLUE_OCEAN);
}

std::string ModernShell::GradientSystem::apply_gradient(const std::string& text, GradientType type, bool smooth) const {
    const auto& preset = get_preset(type);
    if (text.empty() || preset.colors.empty()) {
        return text;
    }

    std::string result;
    const auto& colors = preset.colors;
    size_t color_count = colors.size();
    if (smooth) {
        // Smooth gradient - interpolate between colors
        for (size_t i = 0; i < text.length(); ++i) {
            double ratio = static_cast<double>(i) / std::max(1.0, static_cast<double>(text.length() - 1));
            size_t color_index = static_cast<size_t>(ratio * (color_count - 1));
            result += std::string(colors[color_index]) + text[i];
        }
    } else {
        // Segmented gradient
        size_t segment_length = std::max(size_t(1), text.length() / color_count);
        for (size_t i = 0; i < text.length(); ++i) {
            size_t color_index = (i / segment_length) % color_count;
                       result += std::string(colors[color_index]) + text[i];
        }
    }

    result += esql::colors::RESET;
    return result;
}

std::string ModernShell::GradientSystem::apply_vertical_gradient(const std::vector<std::string>& lines, GradientType type) const {
    const auto& preset = get_preset(type);
    if (lines.empty() || preset.colors.empty()) {
        return "";
    }

    std::string result;
    const auto& colors = preset.colors;
    size_t color_count = colors.size();

    for (size_t line_idx = 0; line_idx < lines.size(); ++line_idx) {
        double ratio = static_cast<double>(line_idx) / std::max(1.0, static_cast<double>(lines.size() - 1));
        size_t color_index = static_cast<size_t>(ratio * (color_count - 1));
        result += std::string(colors[color_index]) + lines[line_idx] + esql::colors::RESET + "\n";
    }

    return result;
}

// Helper gradient methods
std::string ModernShell::apply_text_gradient(const std::string& text, const std::vector<const char*>& colors, bool smooth) const {
    if (colors.empty() || text.empty()) {
        return text;
    }

    std::string result;
    size_t color_count = colors.size();

    if (smooth) {
        for (size_t i = 0; i < text.length(); ++i) {
            double ratio = static_cast<double>(i) / std::max(1.0, static_cast<double>(text.length() - 1));
            size_t color_index = static_cast<size_t>(ratio * (color_count - 1));
            result += std::string(colors[color_index]) + text[i];
        }
    } else {
        size_t segment_length = std::max(size_t(1), text.length() / color_count);
        for (size_t i = 0; i < text.length(); ++i) {
            size_t color_index = (i / segment_length) % color_count;
            result += std::string(colors[color_index]) + text[i];
        }
    }

    result += esql::colors::RESET;
    return result;
}

std::string ModernShell::apply_character_gradient(const std::string& text, const std::vector<const char*>& colors) const {
    if (colors.empty() || text.empty()) {
        return text;
    }

    std::string result;
    size_t color_count = colors.size();

    for (size_t i = 0; i < text.length(); ++i) {
        size_t color_index = i % color_count;
        result += std::string(colors[color_index]) + text[i];
    }

    result += esql::colors::RESET;
    return result;
}

void ModernShell::print_banner() {
    terminal_.clear_screen();

    if (use_colors_) {
        // Position cursor at top
        std::cout << "\033[1;1H";

        // Create Phoenix animator
        PhoenixAnimator phoenix_animator(terminal_width_);

        // ESQL logo (6 lines) - Apply gradient to logo only
        std::vector<std::string> esql_logo = {
            "   ███████╗███████╗ ██████╗ ██╗   ",
            "   ██╔════╝██╔════╝██╔═══██╗██║   ",
            "   █████╗  ███████╗██║   ██║██║   ",
            "   ██╔══╝  ╚════██║██║   ██║██║   ",
            "   ███████╗███████║╚██████╔╝███████╗",
            "   ╚══════╝╚══════╝ ╚═════╝ ╚══════╝"
        };

        // Apply gradient to ESQL logo
        std::vector<std::string> gradient_logo = esql_logo;
        for (size_t i = 0; i < gradient_logo.size(); ++i) {
            gradient_logo[i] = gradient_system_.apply_gradient(
                gradient_logo[i], 
                GradientSystem::GradientType::CYAN_AURORA, 
                false  // Segmented gradient for better readability
            );
        }

        int esql_height = esql_logo.size();
        int phoenix_height = phoenix_animator.get_current_frame().size();

        // Calculate positioning - align bases with 0 free space
        int phoenix_start_row = 1; // Phoenix starts at row 1
        int esql_start_row = 1 + (phoenix_height - esql_height); // ESQL starts at row 14 (1 + 13)

        int phoenix_start_col = terminal_width_ - 45;
        if (phoenix_start_col < 40) phoenix_start_col = 40;

        // Draw Phoenix art (19 lines starting from row 1) - NO GRADIENT (keep original fire effect)
        for (int i = 0; i < phoenix_height; ++i) {
            std::cout << "\033[" << (phoenix_start_row + i) << ";" << phoenix_start_col << "H";
            std::string colored_line = phoenix_animator.apply_gradient(
                phoenix_animator.get_current_frame()[i], i * 2);
            std::cout << colored_line;
        }

        // Draw ESQL logo with gradient (6 lines starting from row 14)
        for (int i = 0; i < esql_height; ++i) {
            std::cout << "\033[" << (esql_start_row + i) << ";1H";
            std::cout << gradient_logo[i]; // Use gradient version
        }

        // Header box with gradient
        int header_start_line = phoenix_start_row + phoenix_height + 1; // 1 + 19 + 1 = 21
        std::cout << "\033[" << header_start_line << ";1H";

        // Apply gradient to header text only (not the box borders)
        std::string title_line1 = "    E N H A N C E D   ES Q L   S H E L L  ";
        std::string title_line2 = "        H4CK3R  STYL3  V3RSI0N         ";
        
        std::string gradient_title1 = gradient_system_.apply_gradient(
            title_line1, GradientSystem::GradientType::PURPLE_DAWN, true);
        std::string gradient_title2 = gradient_system_.apply_gradient(
            title_line2, GradientSystem::GradientType::ORANGE_SUNSET, true);

        std::cout << esql::colors::CYAN << "╔═══════════════════════════════════════╗\n";
        std::cout << "║" << gradient_title1 << esql::colors::CYAN << "║\n";
        std::cout << "║" << gradient_title2 << esql::colors::CYAN << "║\n";
        std::cout << "╚═══════════════════════════════════════╝\n" << esql::colors::RESET;

        // Status messages with subtle gradients
        int status_start_line = header_start_line + 4; // 21 + 4 = 25
        std::cout << "\033[" << status_start_line << ";1H";

        // Apply subtle gradients to status messages
        std::vector<std::string> status_messages = {
            "Type 'help' for commands, 'exit' to quit",
            "Initializing ESQL Database Matrix...",
            "Quantum ESQL Processor: ONLINE",
            "Syntax Highlighting: ACTIVATED"
        };

        for (size_t i = 0; i < status_messages.size(); ++i) {
            std::cout << esql::colors::RED << "[*] ";
            std::string gradient_msg = gradient_system_.apply_gradient(
                status_messages[i], 
                GradientSystem::GradientType::BLUE_OCEAN, 
                true
            );
            std::cout << gradient_msg << esql::colors::RESET << "\n";
        }

        std::cout.flush();

        // STEP 1: Phoenix fire effect comes to life for 3 seconds after "Syntax Highlighting"
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::thread phoenix_thread([&phoenix_animator, phoenix_start_row, phoenix_start_col]() {
            phoenix_animator.animate_fire_effect(3000);
        });
        phoenix_thread.join();

        // STEP 2: First line animation - NO GRADIENT (keep original animation)
        int anim1_line = status_start_line + 4;
        std::cout << "\033[" << anim1_line << ";1H";
        std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::RESET;
        std::cout.flush();

        ConsoleAnimator animator1(terminal_width_);
        animator1.animateText("Forged from the fires of performance for the warriors of the digital age", 4000);

        // STEP 3: Second line animation - NO GRADIENT (keep original wave animation)
        int anim2_line = anim1_line + 1;
        std::cout << "\033[" << anim2_line << ";1H";
        std::cout << esql::colors::MAGENTA << "[+] " << esql::colors::RESET;
        std::cout.flush();

        WaveAnimator animator2(terminal_width_);
        animator2.waveAnimation("accessing the esql framework console", 2);

        // STEP 4: Final connection line with subtle gradient
        int conn_line = anim2_line + 1;
        std::cout << "\033[" << conn_line << ";1H";
        
        std::string connection_text = "Connected to: " + current_db_ + " ●";
        std::string gradient_connection = gradient_system_.apply_gradient(
            connection_text, 
            GradientSystem::GradientType::GREEN_FOREST, 
            true
        );
        
        std::cout << esql::colors::MAGENTA << "[+] " << gradient_connection 
                  << esql::colors::RESET << "\n\n";

    } else {
        // Fallback for no colors
        std::cout << "ESQL SHELL - Enhanced Query Language Shell\n";
        std::cout << "Connected to: " << current_db_ << "\n\n";
    }
}

/*void ModernShell::print_banner() {
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
}*/

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
        // Apply gradient to time portion only
        std::string gradient_time = gradient_system_.apply_gradient(
            "[" + time_str + "]",
            GradientSystem::GradientType::BLUE_OCEAN,
            false
        );

        prompt = gradient_time + " " +
                 std::string(esql::colors::GREEN) + "• " +
                 esql::colors::RESET +
                 std::string(esql::colors::GRAY) + current_db_ +
                 esql::colors::RESET + "> ";
    } else {
        prompt = "[" + time_str + "] • " + current_db_ + "> ";
    }

    return prompt;
}

/*std::string ModernShell::build_prompt() const {
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
}*/

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

    // Calculate minimum reasonable widths for each column
    std::vector<size_t> widths(result.columns.size());
    for (size_t i = 0; i < result.columns.size(); ++i) {
        // Start with column header width + padding
        widths[i] = result.columns[i].length() + 2;

        // Calculate reasonable content width
        size_t max_content_width = 0;
        for (const auto& row : result.rows) {
            if (i < row.size() && !row[i].empty()) {
                // For content, use smarter width calculation
                size_t reasonable_width = calculate_reasonable_content_width(row[i]);
                if (reasonable_width > max_content_width) {
                    max_content_width = reasonable_width;
                }
            }
        }

        // Use the larger of header width or reasonable content width
        widths[i] = std::max(widths[i], max_content_width + 2);

        // Set reasonable maximums based on data type and importance
        widths[i] = apply_smart_maximums(result.columns[i], widths[i]);
    }

    // Ensure we don't exceed terminal width
    size_t total_width = std::accumulate(widths.begin(), widths.end(), 0) + (widths.size() - 1);
    if (total_width > static_cast<size_t>(terminal_width_)) {
        // Scale down proportionally
        double scale_factor = static_cast<double>(terminal_width_) / total_width;
        for (size_t i = 0; i < widths.size(); ++i) {
            widths[i] = static_cast<size_t>(widths[i] * scale_factor);
            // Ensure minimum width
            widths[i] = std::max(widths[i], result.columns[i].length() + 2);
        }
    }

    // Rest of your existing table drawing code remains the same...
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
            std::string display_value = format_cell_value(row[i], widths[i] - 2);
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

// New helper methods for smart column sizing
size_t ModernShell::calculate_reasonable_content_width(const std::string& value) {
    if (value.empty()) return 4; // "NULL" or empty

    // Common data type patterns with reasonable display widths
    if (is_numeric(value)) {
        // Numbers: show full value up to reasonable limit
        return std::min(value.length(), size_t(12));
    } else if (is_email(value)) {
        // Emails: show reasonable portion (username@domain...)
        return std::min(value.length(), size_t(20));
    } else if (is_date(value)) {
        // Dates: full date format
        return std::min(value.length(), size_t(12));
    } else if (is_boolean(value)) {
        // Boolean: "TRUE"/"FALSE"
        return 5;
    } else if (value.length() <= 15) {
        // Short strings: show full
        return value.length();
    } else {
        // Long strings: reasonable default
        return std::min(value.length(), size_t(25));
    }
}

size_t ModernShell::apply_smart_maximums(const std::string& column_name, size_t current_width) {
    std::string lower_name = column_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    // Set maximums based on column name patterns
    if (lower_name.find("id") != std::string::npos) {
        return std::min(current_width, size_t(15));
    } else if (lower_name.find("name") != std::string::npos) {
        return std::min(current_width, size_t(20));
    } else if (lower_name.find("email") != std::string::npos) {
        return std::min(current_width, size_t(25));
    } else if (lower_name.find("salary") != std::string::npos ||
               lower_name.find("amount") != std::string::npos ||
               lower_name.find("price") != std::string::npos) {
        return std::min(current_width, size_t(12)); // Numbers
    } else if (lower_name.find("date") != std::string::npos ||
               lower_name.find("time") != std::string::npos) {
        return std::min(current_width, size_t(12));
    } else if (lower_name.find("desc") != std::string::npos ||
               lower_name.find("note") != std::string::npos ||
               lower_name.find("comment") != std::string::npos) {
        return std::min(current_width, size_t(30));
    } else {
        // Default maximum
        return std::min(current_width, size_t(25));
    }
}

// Improved format_cell_value to show more content
std::string ModernShell::format_cell_value(const std::string& value, size_t max_width) {
    if (value.empty()) return "NULL";
    if (value.length() <= max_width) return value;

    // For numeric values, try to show more digits
    if (is_numeric(value) && max_width >= 8) {
        // For numbers, prefer to show beginning (most significant digits)
        return value.substr(0, max_width - 3) + "...";
    }

    // For emails, show username part fully if possible
    if (is_email(value)) {
        size_t at_pos = value.find('@');
        if (at_pos != std::string::npos && at_pos <= max_width - 4) {
            return value.substr(0, max_width - 3) + "...";
        }
    }

    // For other values, use smarter truncation
    if (max_width >= 12) {
        // Show both beginning and end for medium widths
        size_t first_part = max_width * 2 / 3;
        size_t last_part = max_width - first_part - 3;

        if (last_part >= 3 && first_part >= 3) {
            return value.substr(0, first_part) + "..." + value.substr(value.length() - last_part);
        }
    }

    // Simple truncation as fallback
    return value.substr(0, max_width - 3) + "...";
}

// Additional pattern detection helpers
bool ModernShell::is_numeric(const std::string& value) {
    if (value.empty()) return false;

    // Check if it's a number (integer or decimal)
    bool has_digit = false;
    bool has_decimal = false;

    for (char c : value) {
        if (std::isdigit(c)) {
            has_digit = true;
        } else if (c == '.' && !has_decimal) {
            has_decimal = true;
        } else if (c == '-' || c == '+') {
            // Allow signs only at beginning
            if (&c != &value[0]) return false;
        } else {
            return false;
        }
    }

    return has_digit;
}

bool ModernShell::is_email(const std::string& value) {
    return value.find('@') != std::string::npos &&
           value.find('.') != std::string::npos &&
           value.length() > 5;
}

bool ModernShell::is_date(const std::string& value) {
    // Simple date pattern detection (YYYY-MM-DD or similar)
    static const std::regex date_pattern("^\\d{4}-\\d{2}-\\d{2}");
    return std::regex_search(value, date_pattern);
}

bool ModernShell::is_boolean(const std::string& value) {
    std::string upper_value = value;
    std::transform(upper_value.begin(), upper_value.end(), upper_value.begin(), ::toupper);
    return upper_value == "TRUE" || upper_value == "FALSE" ||
           upper_value == "1" || upper_value == "0" ||
           upper_value == "T" || upper_value == "F";
}

/*void ModernShell::print_results(const ExecutionEngine::ResultSet& result, double duration) {
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
}*/

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
