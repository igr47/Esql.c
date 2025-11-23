#ifndef SYNTAX_HIGHLIGHTER_H
#define SYNTAX_HIGHLIGHTER_H

#include "shell_types.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace esql {

class SyntaxHighlighter {
public:
    SyntaxHighlighter();
    
    // Main highlighting function
    std::string highlight(const std::string& input);
    
    // Configuration
    void enable_colors(bool enable) { use_colors_ = enable; }
    void set_current_database(const std::string& db_name) { current_db_ = db_name; }
    
private:
    // SQL language definitions
    static const std::unordered_set<std::string> keywords;
    static const std::unordered_set<std::string> datatypes;
    static const std::unordered_set<std::string> constraints;
    static const std::unordered_set<std::string> conditionals;
    static const std::unordered_set<std::string> aggregate_functions;
    static const std::unordered_set<std::string> operators;
    
    // Token processing
    void process_word(const std::string& word, std::string& result);
    void process_token(const std::string& token, std::string& result);
    
    // Color application
    std::string apply_color(const std::string& text, const char* color_code);
    
    // State tracking
    struct ParseState {
        bool in_string = false;
        bool in_number = false;
        bool in_comment = false;
        char string_delim = 0;
        std::string current_word;
        std::string current_number;
    };
    
    bool use_colors_ = true;
    std::string current_db_;
};

} // namespace esql

#endif
