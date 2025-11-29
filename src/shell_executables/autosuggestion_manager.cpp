#include "shell_includes/autosuggestion_manager.h"
#include <algorithm>
#include <cctype>

namespace esql {

AutoSuggestionManager::AutoSuggestionManager(HistoryManager& history) 
    : history_(history) {}

AutoSuggestion AutoSuggestionManager::get_suggestion(const std::string& current_input) {
    AutoSuggestion suggestion;
    
    if (!enabled_ || current_input.empty()) {
        return suggestion;
    }
    
    std::string best_match = find_best_match(current_input);
    
    if (!best_match.empty() && best_match != current_input) {
        suggestion.suggestion = best_match;
        suggestion.prefix = current_input;
        suggestion.active = true;
        suggestion.display_start = current_input.length();
    }
    
    return suggestion;
}

std::string AutoSuggestionManager::find_best_match(const std::string& prefix) {
    if (prefix.empty()) return "";
    
    const auto& history = history_.get_all();
    
    // Search from most recent to oldest (reverse iteration)
    // This mimics zsh behavior - most recent matching command gets suggested
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        const std::string& command = *it;
        
        // Check if this command starts with the current input prefix
        if (command.length() > prefix.length() && 
            command.compare(0, prefix.length(), prefix) == 0) {
            return command;
        }
        
        // Also check for case-insensitive match (optional, but user-friendly)
        std::string command_lower = command;
        std::string prefix_lower = prefix;
        std::transform(command_lower.begin(), command_lower.end(), command_lower.begin(), ::tolower);
        std::transform(prefix_lower.begin(), prefix_lower.end(), prefix_lower.begin(), ::tolower);
        
        if (command_lower.length() > prefix_lower.length() && 
            command_lower.compare(0, prefix_lower.length(), prefix_lower) == 0) {
            return command;
        }
    }
    
    return "";
}

std::string AutoSuggestionManager::accept_suggestion(const std::string& current_input, 
                                                   const AutoSuggestion& suggestion) {
    if (!suggestion.active || current_input != suggestion.prefix) {
        return current_input;
    }
    
    return suggestion.suggestion;
}

int AutoSuggestionManager::calculate_similarity(const std::string& str1, const std::string& str2) {
    // This function is no longer used in the primary matching logic
    // but kept for potential future use
    
    // Simple scoring: length-based with recency bonus
    int score = str2.length(); // Longer commands get higher score
    
    // Exact prefix match bonus
    if (str2.find(str1) == 0) {
        score += 100;
    }
    
    return score;
}

} // namespace esql
