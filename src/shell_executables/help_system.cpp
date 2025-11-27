#include "help_system.h"
#include <iostream>
#include <iomanip>

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"
#define COLOR_BOLD    "\033[1m"

void HelpSystem::printHeader(const std::string& title) {
    std::cout << COLOR_CYAN << COLOR_BOLD << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::setw(58) << std::left << title << " ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << COLOR_RESET << "\n\n";
}

void HelpSystem::printQueryExample(const std::string& syntax, const std::string& description) {
    std::cout << COLOR_GREEN << "  " << syntax << COLOR_RESET << "\n";
    std::cout << COLOR_YELLOW << "    → " << description << COLOR_RESET << "\n\n";
}

void HelpSystem::printNote(const std::string& note) {
    std::cout << COLOR_MAGENTA << "  📝 " << note << COLOR_RESET << "\n\n";
}

const std::unordered_map<std::string, std::string> HelpSystem::categoryDescriptions = {
    {"DATABASE", "Database creation, selection and management commands"},
    {"TABLE", "Table structure operations (CREATE, ALTER, DROP)"},
    {"DATA_QUERY", "Data retrieval and querying commands"},
    {"DATA_MANIPULATION", "Data modification commands (INSERT, UPDATE, DELETE)"},
    {"BULK_OPERATIONS", "Batch operations for performance"},
    {"SCHEMA", "Database and table structure inspection"}
};

const std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> HelpSystem::queryExamples = {
    // DATABASE OPERATIONS
    {"DATABASE", {
        {"CREATE DATABASE dbname", "Create a new database"},
        {"USE DATABASE dbname", "Switch to a specific database"},
        {"USE dbname", "Alternative syntax for switching database"},
        {"SHOW DATABASES", "List all available databases"}
    }},
    
    // TABLE OPERATIONS  
    {"TABLE", {
        {"CREATE TABLE table_name (col1 INT, col2 TEXT, col3 BOOL)", "Create table with basic columns"},
        {"CREATE TABLE table_name (id INT PRIMARY_KEY AUTO_INCREAMENT, name TEXT NOT_NULL)", "Create table with constraints"},
        {"CREATE TABLE table_name (created_date DATE GENERATE_DATE, uuid_col UUID GENERATE_UUID)", "Create table with auto-generated values"},
        {"CREATE TABLE table_name (age INT CHECK(age >= 0 AND age <= 150))", "Create table with check constraints"},
        {"DROP TABLE table_name", "Remove a table completely"},
        {"ALTER TABLE table_name ADD new_col INT", "Add new column to table"},
        {"ALTER TABLE table_name ADD new_col TEXT NOT_NULL UNIQUE", "Add column with constraints"},
        {"ALTER TABLE table_name DROP column_name", "Remove column from table"},
        {"ALTER TABLE table_name RENAME old_col TO new_col", "Rename existing column"}
    }},
    
    // DATA QUERY OPERATIONS
    {"DATA_QUERY", {
        {"SELECT * FROM table_name", "Select all columns from table"},
        {"SELECT col1, col2 FROM table_name", "Select specific columns"},
        {"SELECT DISTINCT col1 FROM table_name", "Select unique values only"},
        {"SELECT COUNT(*) FROM table_name", "Count all rows"},
        {"SELECT COUNT(column_name) FROM table_name", "Count non-null values"},
        {"SELECT SUM(column_name) FROM table_name", "Calculate sum of numeric column"},
        {"SELECT AVG(column_name) FROM table_name", "Calculate average value"},
        {"SELECT MIN(column_name), MAX(column_name) FROM table_name", "Find min and max values"},
        {"SELECT column_name AS alias FROM table_name", "Use column aliases"},
        {"SELECT * FROM table_name WHERE condition", "Filter rows with WHERE clause"},
        {"SELECT * FROM table_name WHERE col1 = 'value'", "Equality condition"},
        {"SELECT * FROM table_name WHERE col1 LIKE 'A%'", "Pattern matching with LIKE"},
        {"SELECT * FROM table_name WHERE col1 BETWEEN 10 AND 20", "Range condition"},
        {"SELECT * FROM table_name WHERE col1 IN ('val1', 'val2')", "Membership condition"},
        {"SELECT * FROM table_name WHERE col1 IS NULL", "Check for NULL values"},
        {"SELECT * FROM table_name WHERE col1 IS NOT NULL", "Check for non-NULL values"},
        {"SELECT * FROM table_name WHERE col1 IS TRUE", "Boolean condition check"},
        {"SELECT * FROM table_name ORDER BY col1 ASC", "Sort ascending"},
        {"SELECT * FROM table_name ORDER BY col1 DESC", "Sort descending"},
        {"SELECT * FROM table_name ORDER BY col1, col2 DESC", "Multiple column sorting"},
        {"SELECT * FROM table_name LIMIT 10", "Limit result rows"},
        {"SELECT * FROM table_name LIMIT 10 OFFSET 5", "Limit with offset (pagination)"},
        {"SELECT * FROM table_name GROUP BY col1", "Group rows by column"},
        {"SELECT col1, COUNT(*) FROM table_name GROUP BY col1", "Aggregate with grouping"},
        {"SELECT col1, COUNT(*) FROM table_name GROUP BY col1 HAVING COUNT(*) > 5", "Filter groups with HAVING"},
        {"SELECT CASE WHEN condition THEN result ELSE default END FROM table_name", "Conditional logic with CASE"},
        {"SELECT ROUND(column_name, 2) FROM table_name", "Numeric rounding"},
        {"SELECT LOWER(column_name) FROM table_name", "Convert to lowercase"},
        {"SELECT UPPER(column_name) FROM table_name", "Convert to uppercase"},
        {"SELECT SUBSTRING(column_name, 1, 5) FROM table_name", "Extract substring"}
    }},
    
    // DATA MANIPULATION OPERATIONS
    {"DATA_MANIPULATION", {
        {"INSERT INTO table_name VALUES ('val1', 'val2', 'val3')", "Insert single row"},
        {"INSERT INTO table_name (col1, col2) VALUES ('val1', 'val2')", "Insert with specified columns"},
        {"INSERT INTO table_name VALUES ('val1', 'val2'), ('val3', 'val4')", "Insert multiple rows"},
        {"UPDATE table_name SET col1 = 'new_value'", "Update all rows"},
        {"UPDATE table_name SET col1 = 'new_value' WHERE condition", "Update filtered rows"},
        {"UPDATE table_name SET col1 = expr1, col2 = expr2 WHERE condition", "Update multiple columns"},
        {"DELETE FROM table_name", "Delete all rows (dangerous!)"},
        {"DELETE FROM table_name WHERE condition", "Delete filtered rows"}
    }},
    
    // BULK OPERATIONS
    {"BULK_OPERATIONS", {
        {"BULK INSERT INTO table_name VALUES ('val1', 'val2'), ('val3', 'val4')", "Bulk insert multiple rows"},
        {"BULK INSERT INTO table_name (col1, col2) VALUES ('v1', 'v2'), ('v3', 'v4')", "Bulk insert with columns"},
        {"BULK UPDATE table_name SET ROW 1 col1 = 'val1', ROW 2 col1 = 'val2'", "Update multiple specific rows"},
        {"BULK DELETE FROM table_name WHERE ROW IN (1, 3, 5)", "Delete multiple specific rows by ID"}
    }},
    
    // SCHEMA INSPECTION
    {"SCHEMA", {
        {"SHOW TABLES", "List all tables in current database"},
        {"SHOW TABLE STRUCTURE table_name", "Show detailed table schema"},
        {"SHOW DATABASE STRUCTURE dbname", "Show database structure and statistics"}
    }}
};

void HelpSystem::showMainHelp() {
    printHeader("ESQL DATABASE SYSTEM - COMMAND REFERENCE");
    
    std::cout << COLOR_WHITE << COLOR_BOLD << "Available Command Categories:\n" << COLOR_RESET;
    std::cout << "─────────────────────────────────────────────────────────────────────\n";
    
    for (const auto& category : getAvailableCategories()) {
        auto it = categoryDescriptions.find(category);
        std::string desc = (it != categoryDescriptions.end()) ? it->second : "No description available";
        std::cout << COLOR_GREEN << "  " << std::setw(20) << std::left << category 
                  << COLOR_RESET << " - " << desc << "\n";
    }
    
    std::cout << "\n" << COLOR_YELLOW << "Usage: " << COLOR_RESET;
    std::cout << COLOR_CYAN << "HELP <category>" << COLOR_RESET << " or " << COLOR_CYAN << "HELP <command>" << COLOR_RESET << "\n";
    std::cout << COLOR_YELLOW << "Examples: " << COLOR_RESET;
    std::cout << COLOR_CYAN << "HELP SELECT" << COLOR_RESET << ", " << COLOR_CYAN << "HELP DATA_QUERY" << COLOR_RESET << "\n\n";
    
    printNote("Use uppercase for SQL keywords. Table and column names are case-sensitive.");
}

void HelpSystem::showCategoryHelp(const std::string& category) {
    auto it = queryExamples.find(category);
    if (it == queryExamples.end()) {
        std::cout << COLOR_RED << "Unknown category: " << category << COLOR_RESET << "\n";
        showMainHelp();
        return;
    }
    
    printHeader("ESQL - " + category + " COMMANDS");
    
    auto descIt = categoryDescriptions.find(category);
    if (descIt != categoryDescriptions.end()) {
        std::cout << COLOR_WHITE << descIt->second << COLOR_RESET << "\n\n";
    }
    
    for (const auto& example : it->second) {
        printQueryExample(example.first, example.second);
    }
    
    std::cout << COLOR_MAGENTA << "Use 'HELP <specific_command>' for more detailed examples.\n" << COLOR_RESET;
}

void HelpSystem::showQueryHelp(const std::string& queryType) {
    std::string upperQuery = queryType;
    std::transform(upperQuery.begin(), upperQuery.end(), upperQuery.begin(), ::toupper);
    
    // Search through all categories for this query type
    bool found = false;
    
    for (const auto& category : queryExamples) {
        for (const auto& example : category.second) {
            // Extract the first word from the example (the command)
            std::string firstWord = example.first.substr(0, example.first.find(' '));
            std::transform(firstWord.begin(), firstWord.end(), firstWord.begin(), ::toupper);
            
            if (firstWord == upperQuery) {
                if (!found) {
                    printHeader("ESQL - " + upperQuery + " COMMAND DETAILS");
                    found = true;
                }
                
                printQueryExample(example.first, example.second);
            }
        }
    }
    
    if (!found) {
        std::cout << COLOR_RED << "No detailed help found for: " << queryType << COLOR_RESET << "\n";
        showMainHelp();
    }
}

std::vector<std::string> HelpSystem::getAvailableCategories() {
    return {"DATABASE", "TABLE", "DATA_QUERY", "DATA_MANIPULATION", "BULK_OPERATIONS", "SCHEMA"};
}
