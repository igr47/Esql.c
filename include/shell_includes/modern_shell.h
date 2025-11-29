#ifndef MODERN_SHELL_H
#define MODERN_SHELL_H

#include "shell_types.h"
#include "terminal_input.h"
#include "history_manager.h"
#include "database.h"
#include "utf8_processor.h"
#include "syntax_highlighter.h"
#include "animator.h"
#include "completion_engine.h"
#include "autosuggestion_manager.h"
#include "phoenix_animator.h"
#include "spell_checker.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

//static const int MIN_FREE_LINES = 3;
//static const int SCROLL_LINES = 5;

static const int MIN_FREE_LINES = 2;
static const int SCROLL_LINES = 2;

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

    class GradientSystem {
        public:
            enum class GradientType {
                BLUE_OCEAN,
                PURPLE_DAWN,
                CYAN_AURORA,
                GREEN_FOREST,
                ORANGE_SUNSET,
                MAGENTA_NEBULA,
                PREMIUM_GOLD
            };

            struct GradientPreset {
                std::vector<const char*> colors;
                std::string name;
            };

            std::unordered_map<GradientType, GradientPreset> presets;

            GradientSystem();
            const GradientPreset& get_preset(GradientType type) const;
            std::string apply_gradient(const std::string& text, GradientType type, bool smooth = true) const;
            std::string apply_vertical_gradient(const std::vector<std::string>& lines, GradientType type) const;
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
    
    // Helper methods for tab completion
    std::string get_current_word_prefix();
    void show_completions(const std::vector<std::string>& completions);

    // Autosuggestion methods
    void update_autosuggestion();
    void accept_autosuggestion();
    void clear_autosuggestion();
    std::string render_with_suggestion(const std::string& input, const esql::AutoSuggestion& suggestion);

    // Spell checking
    std::string render_with_spell_check(const std::string& input);
    void enable_spell_check(bool enable) { spell_checker_.enable_spell_check(enable); }

    void ensure_input_space();  // Only check space for input area
    void scroll_input_area(int lines_to_scroll);   // Scroll only when absolutely necessary

    // Command execution
    void execute_command(const std::string& command);

    void update_prompt_position();
    void move_to_prompt_position();

    int calculate_rendered_lines(const std::string& input, const std::string& prompt);
    void clear_previous_lines(int previous_lines, int current_lines);
    
    // UI components
    void print_banner();
    std::string build_prompt() const;
    void print_prompt();
    void show_help();
    void clear_screen();
    void print_results(const ExecutionEngine::ResultSet& result, double duration);
    void print_structure_results(const ExecutionEngine::ResultSet& result, double duration);

        // Smart column sizing helpers
    size_t calculate_reasonable_content_width(const std::string& value);
    size_t apply_smart_maximums(const std::string& column_name, size_t current_width);
    bool is_numeric(const std::string& value);
    bool is_email(const std::string& value);
    bool is_date(const std::string& value);
    bool is_boolean(const std::string& value);
    std::string format_cell_value(const std::string& value, size_t max_width);

    // Utility
    std::string get_current_time() const;
    esql::KeyCode convert_char_to_keycode(char c);
    esql::KeyCode handle_escape_sequence();
    bool handle_possible_resize();

    std::string apply_text_gradient(const std::string& text, const std::vector<const char*>& colors, bool smooth = true) const;
    std::string apply_character_gradient(const std::string& text, const std::vector<const char*>& colors) const;
    void print_gradient_banner();
    
    // Core components
    Database& db_;
    esql::TerminalInput terminal_;
    esql::HistoryManager history_;
    esql::SyntaxHighlighter highlighter_;
    esql::UTF8Processor utf8_processor_;
    esql::CompletionEngine completion_engine_;
    esql::AutoSuggestionManager autosuggestion_manager_;
    esql::SpellChecker spell_checker_;
    GradientSystem gradient_system_;

    // Autosuggestion state
    esql::AutoSuggestion current_suggestion_;
    
    // Fish-like screen state - SIMPLIFIED APPROACH
    std::string current_prompt_;
    std::string last_rendered_input_;
    size_t last_cursor_pos_ = 0;
   
    bool in_multiline_mode_ = false;
    std::vector<std::string> multiline_buffer_;

    int prompt_row_ = 0;  // Current prompt row position
    int prompt_col_ = 1;  // Current prompt column position
    int previous_render_lines_ = 1;   // Track previous render lines for proper clearing

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
