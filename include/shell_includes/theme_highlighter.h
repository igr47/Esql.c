#ifndef THEME_HIGHLIGHTER_H
#define THEME_HIGHLIGHTER_H

#include "theme_system.h"
#include "keyword_groups.h"
#include "theme_highlighter.h"
#include <string>
#include <vector>

namespace esql {

class ThemeHighlighter {
public:
    ThemeHighlighter(ThemeSystem& theme_system);
    
    // Highlight SQL code using current theme
    std::string highlight(const std::string& input);
    
    // Highlight with specific theme
    std::string highlight_with_theme(const std::string& input, const ThemeSystem::Theme& theme);
    
    // Configuration
    void enable_colors(bool enable) { enabled_ = enable; }
    bool is_enabled() const { return enabled_; }
    
    // Set current database for context-aware highlighting
    void set_current_database(const std::string& db_name) { current_db_ = db_name; }
    
private:
    ThemeSystem& theme_system_;
    bool enabled_ = true;
    std::string current_db_;
    
    // Tokenization
    struct Token {
        std::string text;
        size_t start;
        size_t end;
        std::string type; // keyword, string, number, comment, etc.
        std::string keyword_group; // For keywords
    };
    
    // Parse SQL into tokens
    std::vector<Token> tokenize(const std::string& input);
    
    // Determine token type
    std::string get_token_type(const std::string& token, size_t position, const std::string& context);
    
    // Apply theme style to token
    std::string apply_theme_style(const Token& token, const ThemeSystem::Theme& theme);
    
    // Helper for string and comment handling
    bool is_string_start(char c) const { return c == '\'' || c == '"'; }
    bool is_comment_start(const std::string& input, size_t pos) const;
    bool is_number_start(char c) const;
    bool is_operator(char c) const;
    bool is_punctuation(char c) const;
    
    // Keyword detection
    bool is_keyword(const std::string& word) const;
    std::string get_keyword_group(const std::string& keyword) const;
};

} // namespace esql

#endif // THEME_HIGHLIGHTER_H
