#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>

// Pattern matching methods
bool ExecutionEngine::evaluateCharacterClassMatch(const std::string& str, const std::string& charClassPattern) {
    if (str.empty()) return false;

    // Simple charachter matching
    char c = str[0];
    size_t len = charClassPattern.length();
    size_t i = 0;
    bool negated = false;

    if (i < len && charClassPattern[i] == '^') {
        negated = true;
        i++;
    }

    bool matched = false;
    while (i < len) {
        if (i +2 < len && charClassPattern[i + 1] == '-') {
            // Charachter range
            char start = charClassPattern[i];
            char end = charClassPattern [i + 2];
            if (c >= start && c <= end) {
                matched = true;
                break;
            }
            i += 3;
        } else {
            // Single charachter
            if (c == charClassPattern[i]) {
                matched = true;
                break;
            }
            i++;
        }
    }

    return negated ? !matched : matched;
}

std::string ExecutionEngine::likePatternToRegex(const std::string& likePattern) {
    std::string regex;
    size_t len = likePattern.length();
    size_t i = 0;

    while (i < len) {
        if (likePattern[i] == '%') {
            regex += ".*"; // % matches any sequence of characters
            i++;
        } else if (likePattern[i] == '_') {
            regex += "."; // _ matches any single character
            i++;
        } else if (likePattern[i] == '[' && i + 1 < len) {
            // Handle character class
            i++;
            std::string charClass;
            bool negated = false;

            // Check for negation
            if (i < len && likePattern[i] == '^') {
                negated = true;
                i++;
            }

            // Parse character class contents
            while (i < len && likePattern[i] != ']') {
                if (i + 2 < len && likePattern[i + 1] == '-') {
                    // Handle ranges: A-Z, 0-9
                    charClass += likePattern[i];
                    charClass += '-';
                    charClass += likePattern[i + 2];
                    i += 3;
                } else {
                    // Single character
                    charClass += likePattern[i];
                    i++;
                }
            }

            if (i < len && likePattern[i] == ']') {
                if (negated) {
                    regex += "[^";
                    regex += charClass;
                    regex += "]";
                } else {
                    regex += "[";
                    regex += charClass;
                    regex += "]";
                }
                i++;
            } else {
                // Unclosed bracket, treat as literal
                regex += "\\[";
                if (negated) regex += "\\^";
                regex += charClass;
            }
        } else if (likePattern[i] == '\\' && i + 1 < len) {
            // Escape character
            regex += '\\';
            regex += likePattern[i + 1];
            i += 2;
        } else {
            // Regular character - escape regex special characters
            if (isRegexSpecialChar(likePattern[i])) {
                regex += '\\';
            }
            regex += likePattern[i];
            i++;
        }
    }

    return "^" + regex + "$"; // Match entire string
}

bool ExecutionEngine::isRegexSpecialChar(char c) {
    return c == '.' || c == '^' || c == '$' || c == '*' || c == '+' || c == '?' || c == '|' || c == '(' || c == ')' || c== '{' || c == '}' || c == '\\' || c == '[';
}

std::string ExecutionEngine::expandCharacterClass(const std::string& charClass) {
    std::string expanded;
    size_t len = charClass.length();
    size_t i = 0;

    while (i < len) {
        if (i + 2 < len && charClass[i + 1] == '-') {
            char start = charClass[i];
            char end = charClass[i + 2];

            if (start <= end) {
                // Expand the range
                for (char c = start; c <= end; c++) {
                    expanded += c;
                }
            }
            i += 3;
        } else {
            // Single charachter
            expanded += charClass[i];
            i++;
        }
    }

    return expanded;
}

bool ExecutionEngine::simplePatternMatch(const std::string& str, const std::string& pattern) {
    size_t strPos = 0, patternPos = 0;
    size_t strLen = str.length(), patternLen = pattern.length();
    
    while (patternPos < patternLen && strPos < strLen) {
        if (pattern[patternPos] == '.') {
            // . matches any single character
            if (patternPos + 1 < patternLen && pattern[patternPos + 1] == '*') {
                // .* matches anything - complex case, use regex
                return simpleRegexMatch(str, likePatternToRegex(pattern));
            } else {
                patternPos++;
                strPos++;
            }
        } else if (pattern[patternPos] == '[' && patternPos + 1 < patternLen) {
            // Handle character class
            patternPos++; // Skip '['
            bool matched = false;
            bool negated = false;
            
            // Check for negation
            if (patternPos < patternLen && pattern[patternPos] == '^') {
                negated = true;
                patternPos++;
            }
            
            // Parse character class
            while (patternPos < patternLen && pattern[patternPos] != ']') {
                if (patternPos + 2 < patternLen && pattern[patternPos + 1] == '-') {
                    // Character range
                    char start = pattern[patternPos];
                    char end = pattern[patternPos + 2];
                    
                    if (strPos < strLen && str[strPos] >= start && str[strPos] <= end) {
                        matched = true;
                    }
                    patternPos += 3;
                } else {
                    // Single character
                    if (strPos < strLen && str[strPos] == pattern[patternPos]) {
                        matched = true;
                    }
                    patternPos++;
                }
            }
            
            if (patternPos < patternLen && pattern[patternPos] == ']') {
                patternPos++; // Skip ']'
            }
            
            if (negated) {
                if (matched) return false;
            } else {
                if (!matched) return false;
            }
            strPos++;
        } else if (str[strPos] == pattern[patternPos]) {
            // Regular character match
            patternPos++;
            strPos++;
        } else {
            return false;
        }
    }
    
    return (patternPos >= patternLen && strPos >= strLen);
}

bool ExecutionEngine::simpleRegexMatch(const std::string& str, const std::string& regexPattern) {
    // Remove the ^ and $ anchors for our matching
    std::string pattern = regexPattern;
    bool startsWithAnchor = false;
    bool endsWithAnchor = false;

    if (!pattern.empty() && pattern[0] == '^') {
        startsWithAnchor = true;
        pattern = pattern.substr(1);
    }
    if (!pattern.empty() && pattern[pattern.size()-1] == '$') {
        endsWithAnchor = true;
        pattern = pattern.substr(0, pattern.size()-1);
    }

    // Use recursive matching for proper wildcard handling
    return matchPattern(str, pattern, 0, 0, startsWithAnchor, endsWithAnchor);
}

bool ExecutionEngine::matchPattern(const std::string& str, const std::string& pattern,
                                  size_t strPos, size_t patternPos,
                                  bool startsWithAnchor, bool endsWithAnchor) {
    // If we've consumed all of the pattern
    if (patternPos >= pattern.length()) {
        // If we require matching the entire string and we haven't consumed it all, fail
        if (endsWithAnchor && strPos < str.length()) {
            return false;
        }
        return true;
    }

    // If we've consumed all of the string but not all of the pattern
    if (strPos >= str.length()) {
        // The only way this can match is if the remaining pattern is all wildcards
        for (size_t i = patternPos; i < pattern.length(); i++) {
            if (pattern[i] != '.' || (i + 1 < pattern.length() && pattern[i + 1] != '*')) {
                return false;
            }
        }
        return true;
    }

    // Handle .* sequences (zero or more of any character)
    if (patternPos + 1 < pattern.length() && pattern[patternPos] == '.' && pattern[patternPos + 1] == '*') {
        // Try matching zero characters
        if (matchPattern(str, pattern, strPos, patternPos + 2, startsWithAnchor, endsWithAnchor)) {
            return true;
        }
        // Try matching one or more characters
        for (size_t i = strPos; i < str.length(); i++) {
            if (matchPattern(str, pattern, i + 1, patternPos + 2, startsWithAnchor, endsWithAnchor)) {
                return true;
            }
        }
        return false;
    }

    // Handle . (any single character)
    if (pattern[patternPos] == '.') {
        return matchPattern(str, pattern, strPos + 1, patternPos + 1, startsWithAnchor, endsWithAnchor);
    }

    // Handle escaped characters
    if (pattern[patternPos] == '\\' && patternPos + 1 < pattern.length()) {
        if (str[strPos] == pattern[patternPos + 1]) {
            return matchPattern(str, pattern, strPos + 1, patternPos + 2, startsWithAnchor, endsWithAnchor);
        }
        return false;
    }

    // Handle character classes [abc] or [a-z]
    if (pattern[patternPos] == '[') {
        size_t endBracket = pattern.find(']', patternPos);
        if (endBracket == std::string::npos) {
            return false; // Malformed pattern
        }

        std::string charClass = pattern.substr(patternPos + 1, endBracket - patternPos - 1);
        bool matched = false;
        bool negated = false;
        size_t i = 0;

        if (!charClass.empty() && charClass[0] == '^') {
            negated = true;
            i++;
        }

        while (i < charClass.length()) {
            if (i + 2 < charClass.length() && charClass[i + 1] == '-') {
                // Character range
                if (str[strPos] >= charClass[i] && str[strPos] <= charClass[i + 2]) {
                    matched = true;
                    break;
                }
                i += 3;
            } else {
                // Single character
                if (str[strPos] == charClass[i]) {
                    matched = true;
                    break;
                }
                i++;
            }
        }

        if ((negated && matched) || (!negated && !matched)) {
            return false;
        }

        return matchPattern(str, pattern, strPos + 1, endBracket + 1, startsWithAnchor, endsWithAnchor);
    }

    // Regular character match
    if (str[strPos] == pattern[patternPos]) {
        return matchPattern(str, pattern, strPos + 1, patternPos + 1, startsWithAnchor, endsWithAnchor);
    }

    return false;
}

std::string ExecutionEngine::evaluateLikeOperation(const AST::LikeOp* likeOp, const std::unordered_map<std::string, std::string>& row) {
    std::string left = evaluateExpression(likeOp->left.get(), row);
    std::string right = evaluateExpression(likeOp->right.get(), row);

    // Handle NULL values
    if (left == "NULL" || right == "NULL") {
        return "NULL";
    }

    // Debug output
    std::cout << "DEBUG LIKE: left='" << left << "', pattern='" << right << "'" << std::endl;

    // Convert LIKE pattern to regex
    std::string regexPattern = likePatternToRegex(right);
    std::cout << "DEBUG LIKE: regex pattern='" << regexPattern << "'" << std::endl;

    try {
        bool matches = simpleRegexMatch(left, regexPattern);
        std::cout << "DEBUG LIKE: match result=" << matches << std::endl;
        return matches ? "true" : "false";
    } catch (...) {
        std::cout << "DEBUG LIKE: match failed, returning false" << std::endl;
        return "false";
    }
}
