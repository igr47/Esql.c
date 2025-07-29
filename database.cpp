#include "database.h"
#include <iostream>

Database::Database(const std::string& filename) 
    : storage(std::make_unique<DiskStorage>(filename)), engine(schema, *storage) {}

void Database::execute(const std::string& query) {
    try {
        auto stmt = parseQuery(query);
        
        SematicAnalyzer analyzer(schema);
        analyzer.analyze(stmt);
        
        auto result = engine.execute(std::move(stmt));
        
        // Print results
        for (const auto& col : result.columns) {
            std::cout << col << "\t";
        }
        std::cout << "\n";
        
        for (const auto& row : result.rows) {
            for (const auto& val : row) {
                std::cout << val << "\t";
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

void Database::startInteractive() {
    std::string input;
    std::cout << "ESQL> ";
    while (std::getline(std::cin, input)) {
        if (input == "exit" || input == "quit") break;
        if (!input.empty()) {
            execute(input);
        }
        std::cout << "ESQL> ";
    }
}

std::unique_ptr<AST::Statement> Database::parseQuery(const std::string& query) {
    Lexer lexer(query);
    Parse parser(lexer);
    return parser.parse();
}	
