#ifndef MODERN_SHELL_H
#define MODERN_SHELL_H

#include "shell_types.h"
#include "terminal_input.h"
#include "history_manager.h"
#include "database.h"
#include "utf8_processor.h"
#include "syntax_highlighter.h"
#include "animator.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

class ModernShell {
public:
    explicit ModernShell(Database& db);
    ~ModernShell();
    
    void run();
    void set_current_database(const std::string& db_name);

private:
    // Fish-like screen buffer system - SIMPLIFIED
    class ScreenBuffer {
    public:
        struct Line {
            std::string text;           // Raw text without ANSI
            std::string ansi_text;      // Text with ANSI codes
            bool is_soft_wrapped = false;
            size_t visual_width = 0;
            
            void clear() { 
                text.clear(); 
                ansi_text.clear();
                visual_width = 0;
            }
        };
        
        std::vector<Line> lines;
        int screen_width = 80;
        struct Cursor { 
            int x = 0; 
            int y = 0; 
            Cursor() = default;
            Cursor(int a, int b) : x(a), y(b) {}
        } cursor;
        
        Line& create_line(size_t idx) {
            if (idx >= lines.size()) lines.resize(idx + 1);
            return lines[idx];
        }
        
        Line& line(size_t idx) { 
            if (idx >= lines.size()) lines.resize(idx + 1);
            return lines[idx]; 
        }
        
        size_t line_count() const { return lines.size(); }
        void resize(size_t size) { lines.resize(size); }
        bool empty() const { return lines.empty(); }
        
        void clear() {
            lines.clear();
            cursor.x = 0;
            cursor.y = 0;
        }
    };
    
    // Main loop
    void run_interactive();
    void run_termux_fallback();

    // Fish-like screen management
    void update_screen();
    void reset_abandoning_line();
    void reset_line();
    void refresh_display(bool force_redraw = false);
    
    // Input handling
    void handle_enter();
    void handle_character(char c);
    void handle_backspace();
    void handle_tab();
    void handle_navigation(esql::KeyCode key);
    
    // Command execution
    void execute_command(const std::string& command);

    void update_prompt_position();
    void move_to_prompt_position();
    
    // UI components
    void print_banner();
    std::string build_prompt() const;
    void print_prompt();
    void show_help();
    void clear_screen();
    void print_results(const ExecutionEngine::ResultSet& result, double duration);
    void print_structure_results(const ExecutionEngine::ResultSet& result, double duration);
    
    // Utility
    std::string get_current_time() const;
    esql::KeyCode convert_char_to_keycode(char c);
    esql::KeyCode handle_escape_sequence();
    bool handle_possible_resize();
    
    // Core components
    Database& db_;
    esql::TerminalInput terminal_;
    esql::HistoryManager history_;
    esql::SyntaxHighlighter highlighter_;
    esql::UTF8Processor utf8_processor_;
    
    // Fish-like screen state - SIMPLIFIED APPROACH
    std::string current_prompt_;
    std::string last_rendered_input_;
    size_t last_cursor_pos_ = 0;
   
    bool in_multiline_mode_ = false;
    std::vector<std::string> multiline_buffer_;

    int prompt_row_ = 0;  // Current prompt row position
    int prompt_col_ = 1;  // Current prompt column position

    // State
    std::string current_db_;
    std::string current_input_;
    size_t cursor_pos_ = 0;
    bool should_exit_ = false;
    
    // Configuration
    bool use_colors_ = true;
    int terminal_width_ = 80;
    int terminal_height_ = 24;
};

#endif // MODERN_SHELL_H
