#ifndef KEYWORD_GROUPS_H
#define KEYWORD_GROUPS_H

#include <string>
#include <unordered_set>
#include <vector>

namespace esql {

struct KeywordGroups {
    // Core SQL Keywords
    static const std::unordered_set<std::string> DATA_DEFINITION;
    static const std::unordered_set<std::string> DATA_MANIPULATION;
    static const std::unordered_set<std::string> DATA_QUERY;
    static const std::unordered_set<std::string> DATA_CONTROL;
    
    // Clauses and Modifiers
    static const std::unordered_set<std::string> CLAUSES;
    static const std::unordered_set<std::string> MODIFIERS;
    static const std::unordered_set<std::string> JOIN_KEYWORDS;
    
    // Functions
    static const std::unordered_set<std::string> AGGREGATE_FUNCTIONS;
    static const std::unordered_set<std::string> STRING_FUNCTIONS;
    static const std::unordered_set<std::string> NUMERIC_FUNCTIONS;
    static const std::unordered_set<std::string> DATE_FUNCTIONS;
    static const std::unordered_set<std::string> WINDOW_FUNCTIONS;
    static const std::unordered_set<std::string> STATISTICAL_FUNCTIONS;
    
    // Data Types and Constraints
    static const std::unordered_set<std::string> DATA_TYPES;
    static const std::unordered_set<std::string> CONSTRAINTS;
    
    // Operators and Logical
    static const std::unordered_set<std::string> OPERATORS;
    static const std::unordered_set<std::string> LOGICAL_OPERATORS;
    static const std::unordered_set<std::string> COMPARISON_OPERATORS;
    
    // Conditional and Control Flow
    static const std::unordered_set<std::string> CONDITIONAL;
    static const std::unordered_set<std::string> NULL_HANDLING;
    
    // AI/ML Keywords
    static const std::unordered_set<std::string> AI_CORE;
    static const std::unordered_set<std::string> AI_MODELS;
    static const std::unordered_set<std::string> AI_OPERATIONS;
    static const std::unordered_set<std::string> AI_EVALUATION;
    static const std::unordered_set<std::string> AI_FEATURES;
    static const std::unordered_set<std::string> AI_PREDICTIONS;
    
    // Visualization Keywords
    static const std::unordered_set<std::string> PLOT_TYPES;
    static const std::unordered_set<std::string> GEO_PLOT_TYPES;
    static const std::unordered_set<std::string> PLOT_ELEMENTS;
    static const std::unordered_set<std::string> OUTPUT_FORMATS;
    static const std::unordered_set<std::string> ANIMATION_CONTROLS;
    
    // File Operations
    static const std::unordered_set<std::string> FILE_OPERATIONS;
    
    // System and Utility
    static const std::unordered_set<std::string> SYSTEM_COMMANDS;
    static const std::unordered_set<std::string> UTILITY;
    
    // Special Generators
    static const std::unordered_set<std::string> GENERATORS;
    
    // Get all keywords for a specific group
    static std::vector<std::string> get_group(const std::string& group_name);
    
    // Check if a keyword belongs to a specific group
    static bool is_in_group(const std::string& keyword, const std::string& group_name);
    
    // Get the group name for a keyword
    static std::string get_group_name(const std::string& keyword);
    
    // Get all groups
    static std::vector<std::pair<std::string, std::string>> get_all_groups();
};

} // namespace esql

#endif // KEYWORD_GROUPS_H
