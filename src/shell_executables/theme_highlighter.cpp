#include "shell_includes/theme_highlighter.h"
#include <algorithm>
#include <cctype>
#include <vector>
#include <iostream>

namespace esql {

ThemeHighlighter::ThemeHighlighter(ThemeSystem& theme_system) 
    : theme_system_(theme_system), enabled_(true) {}

std::string ThemeHighlighter::highlight(const std::string& input) {
    if (!enabled_ || input.empty()) {
	//std::cout << "[THEME HIGHLIGHTER] Either the input is empty or not enabled. Detected in highlight()" << std::endl;
        return input;
	//std::cout << "[Theme Highlighter] Returned an empty input." << std::endl;
    }
    
    //std::cout << "[Theme Highlighter] Got input and is enabled. Applyng theme." << std::endl;
    const auto& theme = theme_system_.get_current_theme();
    return highlight_with_theme(input, theme);
    //std::cout << "[Theme Highlighter] applied Highlight with theme." << std::endl;
}

std::string ThemeHighlighter::highlight_with_theme(const std::string& input, const ThemeSystem::Theme& theme) {
    if (input.empty()) return input;
    
    auto tokens = tokenize(input);
    std::string result;
    size_t last_pos = 0;
    
    for (const auto& token : tokens) {
        // Add any text between tokens
        if (token.start > last_pos) {
            result += input.substr(last_pos, token.start - last_pos);
        }
        
        // Apply theme style to token
        result += apply_theme_style(token, theme);
        
        last_pos = token.end;
    }
    
    // Add any remaining text
    if (last_pos < input.length()) {
        result += input.substr(last_pos);
    }
    
    return result;
}

std::vector<ThemeHighlighter::Token> ThemeHighlighter::tokenize(const std::string& input) {
    std::vector<Token> tokens;
    size_t pos = 0;
    size_t length = input.length();
    
    while (pos < length) {
        char c = input[pos];
        
        // Skip whitespace
        if (std::isspace(static_cast<unsigned char>(c))) {
            pos++;
            continue;
        }
        
        // Check for comments
        if (is_comment_start(input, pos)) {
            size_t start = pos;
            // Single line comment
            if (pos + 1 < length && input[pos] == '-' && input[pos + 1] == '-') {
                pos += 2;
                while (pos < length && input[pos] != '\n') {
                    pos++;
                }
                tokens.push_back({input.substr(start, pos - start), start, pos, "comment", ""});
                continue;
            }
            // Multi-line comment
            else if (pos + 1 < length && input[pos] == '/' && input[pos + 1] == '*') {
                pos += 2;
                while (pos < length && !(input[pos] == '*' && pos + 1 < length && input[pos + 1] == '/')) {
                    pos++;
                }
                if (pos < length) pos += 2;
                tokens.push_back({input.substr(start, pos - start), start, pos, "comment", ""});
                continue;
            }
        }
        
        // Check for strings
        if (is_string_start(c)) {
            size_t start = pos;
            char quote = c;
            pos++;
            
            while (pos < length && input[pos] != quote) {
                if (input[pos] == '\\' && pos + 1 < length) {
                    pos += 2; // Skip escaped character
                } else {
                    pos++;
                }
            }
            if (pos < length) pos++; // Skip closing quote
            
            tokens.push_back({input.substr(start, pos - start), start, pos, "string", ""});
            continue;
        }
        
        // Check for numbers
        if (is_number_start(c) || (c == '-' && pos + 1 < length && is_number_start(input[pos + 1]))) {
            size_t start = pos;
            if (c == '-') pos++; // Skip sign
            
            while (pos < length && (std::isdigit(input[pos]) || input[pos] == '.' || 
                   input[pos] == 'e' || input[pos] == 'E' || 
                   (pos > start && (input[pos] == '+' || input[pos] == '-')))) {
                pos++;
            }
            
            tokens.push_back({input.substr(start, pos - start), start, pos, "number", ""});
            continue;
        }
        
        // Check for operators
        if (is_operator(c)) {
            size_t start = pos;
            // Handle multi-character operators (<=, >=, !=, ==, etc.)
            std::string op(1, c);
            if (pos + 1 < length) {
                std::string possible_op = op + input[pos + 1];
                if (possible_op == "<=" || possible_op == ">=" || 
                    possible_op == "!=" || possible_op == "==" ||
                    possible_op == "||" || possible_op == "&&") {
                    pos += 2;
                    op = possible_op;
                } else {
                    pos++;
                }
            } else {
                pos++;
            }
            
            tokens.push_back({op, start, pos, "operator", ""});
            continue;
        }
        
        // Check for punctuation
        if (is_punctuation(c)) {
            tokens.push_back({std::string(1, c), pos, pos + 1, "punctuation", ""});
            pos++;
            continue;
        }
        
        // Handle words/identifiers/keywords
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos;
            while (pos < length && (std::isalnum(static_cast<unsigned char>(input[pos])) || 
                   input[pos] == '_' || input[pos] == '.' || input[pos] == '$')) {
                pos++;
            }
            
            std::string word = input.substr(start, pos - start);
            std::string upper_word = word;
            std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
            
            Token token;
            token.text = word;
            token.start = start;
            token.end = pos;
            
            if (is_keyword(upper_word)) {
                token.type = "keyword";
                token.keyword_group = get_keyword_group(upper_word);
            } else {
                token.type = "identifier";
            }
            
            tokens.push_back(token);
            continue;
        }
        
        // Unknown character, skip it
        pos++;
    }
    
    return tokens;
}


/*std::string ThemeHighlighter::apply_theme_style(const Token& token, const ThemeSystem::Theme& theme) {
    if (token.type == "keyword") {
        // Apply keyword style from theme
        return theme.apply_keyword(token.keyword_group, token.text);
    }

    // Helper function to get style with fallback
    auto get_style = [&](const std::string& element, const std::string& default_style) -> std::string {
        // First check additional_styles map
        auto it = theme.ui_styles.additional_styles.find(element);
        if (it != theme.ui_styles.additional_styles.end()) {
            return it->second;
        }
        return default_style;
    };

    if (token.type == "string") {
        std::string color_name = get_style("string_literal", "string");
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "number") {
        std::string color_name = get_style("number_literal", "number");
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "comment") {
        std::string color_name = get_style("comment", "comment");
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "operator") {
        std::string color_name = get_style("operator", "operator");
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "punctuation") {
        std::string color_name = get_style("punctuation", "punctuation");
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }

    // Default: identifier
    return token.text;
}*/

std::string ThemeHighlighter::apply_theme_style(const Token& token, const ThemeSystem::Theme& theme) {
    //std::cout << "[THEME HIGHLIGHTER] apply_theme_style() - token type: " << token.type 
              //<< ", text: " << token.text << ", group: " << token.keyword_group << std::endl;
    if (token.type == "keyword") {
        // Apply keyword style from theme
	//std::cout << "[Theme Highlighter] Found KEYWORD And now applying theme to key word." << std::endl; 
	//std::cout << "[THEME HIGHLIGHTER] Calling theme.apply_keyword() with group: " << token.keyword_group << ", text: " << token.text << std::endl;
        return theme.apply_keyword(token.keyword_group, token.text);
	//std::cout << "[Theme Highlighter] Applied theme to keyword." << std::endl;
    }

    // Use a proper color mapping for different token types
    std::string color_name;
    
    if (token.type == "string") {
        // Check for additional_styles first, then use get_style
        auto it = theme.ui_styles.additional_styles.find("string_literal");
        if (it != theme.ui_styles.additional_styles.end()) {
            color_name = it->second;
        } else {
            color_name = theme.ui_styles.get_style("string_literal");
        }
        if (color_name.empty()) color_name = "mint"; // Default for Veldora
    }
    else if (token.type == "number") {
        auto it = theme.ui_styles.additional_styles.find("number_literal");
        if (it != theme.ui_styles.additional_styles.end()) {
            color_name = it->second;
        } else {
            color_name = theme.ui_styles.get_style("number_literal");
        }
        if (color_name.empty()) color_name = "coral"; // Default for Veldora
    }
    else if (token.type == "comment") {
        auto it = theme.ui_styles.additional_styles.find("comment");
        if (it != theme.ui_styles.additional_styles.end()) {
            color_name = it->second;
        } else {
            color_name = theme.ui_styles.get_style("comment");
        }
        if (color_name.empty()) color_name = "dark_gray"; // Default for Veldora
    }
    else if (token.type == "operator") {
        auto it = theme.ui_styles.additional_styles.find("operator");
        if (it != theme.ui_styles.additional_styles.end()) {
            color_name = it->second;
        } else {
            color_name = theme.ui_styles.get_style("operator");
        }
        if (color_name.empty()) color_name = "silver"; // Default for Veldora
    }
    else if (token.type == "punctuation") {
        auto it = theme.ui_styles.additional_styles.find("punctuation");
        if (it != theme.ui_styles.additional_styles.end()) {
            color_name = it->second;
        } else {
            color_name = theme.ui_styles.get_style("punctuation");
        }
        if (color_name.empty()) color_name = "light_gray"; // Default for Veldora
    }
    else if (token.type == "identifier") {
        // Get identifier style from theme (table/column names)
        std::string style_value = theme.ui_styles.get_style("identifier");
        if (style_value.empty()) {
            // Fallback to gold (like syntax highlighter)
            style_value = "gold";
        }
        std::string color_code = ColorMapper::name_to_code(style_value);
        return color_code + token.text + colors::RESET_ALL;
    }
    else {
        // Default: identifier
        return token.text;
    }

    // Apply the color
    //std::cout << "[THEME HIGHLIGHTER] Applying color to text. " << std::endl;
    std::string color_code = ColorMapper::name_to_code(color_name);
    return color_code + token.text + colors::RESET_ALL;
    //std::cout << "[Theme Highlighter] Applied colorto text." << std::endl;
}

/*std::string ThemeHighlighter::apply_theme_style(const Token& token, const ThemeSystem::Theme& theme) {
    if (token.type == "keyword") {
        // Apply keyword style from theme
        return theme.apply_keyword(token.keyword_group, token.text);
    }
    else if (token.type == "string") {
        // Apply string style
        std::string color_name = "string";
        auto it = theme.ui_styles.find("string_literal");
        if (it != theme.ui_styles.end()) {
            color_name = it->second;
        }
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "number") {
        // Apply number style
        std::string color_name = "number";
        auto it = theme.ui_styles.find("number_literal");
        if (it != theme.ui_styles.end()) {
            color_name = it->second;
        }
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "comment") {
        // Apply comment style
        std::string color_name = "comment";
        auto it = theme.ui_styles.find("comment");
        if (it != theme.ui_styles.end()) {
            color_name = it->second;
        }
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "operator") {
        // Apply operator style
        std::string color_name = "operator";
        auto it = theme.ui_styles.find("operator");
        if (it != theme.ui_styles.end()) {
            color_name = it->second;
        }
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    else if (token.type == "punctuation") {
        // Apply punctuation style
        std::string color_name = "punctuation";
        auto it = theme.ui_styles.find("punctuation");
        if (it != theme.ui_styles.end()) {
            color_name = it->second;
        }
        std::string color_code = ColorMapper::name_to_code(color_name);
        return color_code + token.text + colors::RESET_ALL;
    }
    
    // Default: identifier
    return token.text;
}*/

bool ThemeHighlighter::is_comment_start(const std::string& input, size_t pos) const {
    if (pos + 1 >= input.length()) return false;
    
    // Check for single line comment
    if (input[pos] == '-' && input[pos + 1] == '-') {
        return true;
    }
    
    // Check for multi-line comment
    if (input[pos] == '/' && input[pos + 1] == '*') {
        return true;
    }
    
    return false;
}

bool ThemeHighlighter::is_number_start(char c) const {
    return std::isdigit(static_cast<unsigned char>(c));
}

bool ThemeHighlighter::is_operator(char c) const {
    static const std::string operators = "=<>!+-*/%&|^~";
    return operators.find(c) != std::string::npos;
}

bool ThemeHighlighter::is_punctuation(char c) const {
    static const std::string punctuation = ",;().[]{}:?@";
    return punctuation.find(c) != std::string::npos;
}

bool ThemeHighlighter::is_keyword(const std::string& word) const {
    // Check all keyword groups
    auto groups = KeywordGroups::get_all_groups();
    for (const auto& [group_name, description] : groups) {
        auto keywords = KeywordGroups::get_group(group_name);
        if (std::find(keywords.begin(), keywords.end(), word) != keywords.end()) {
            return true;
        }
    }
    return false;
}

std::string ThemeHighlighter::get_keyword_group(const std::string& keyword) const {
    return KeywordGroups::get_group_name(keyword);
}
}
