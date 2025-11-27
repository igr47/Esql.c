#ifndef HELP_SYSTEM_H
#define HELP_SYSTEM_H

#include <string>
#include <vector>
#include <unordered_map>

class HelpSystem {
public:
    static void showMainHelp();
    static void showCategoryHelp(const std::string& category);
    static void showQueryHelp(const std::string& queryType);
    static std::vector<std::string> getAvailableCategories();
    
private:
    static void printHeader(const std::string& title);
    static void printQueryExample(const std::string& syntax, const std::string& description);
    static void printNote(const std::string& note);
    
    static const std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> queryExamples;
    static const std::unordered_map<std::string, std::string> categoryDescriptions;
};

#endif
