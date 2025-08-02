#include "database.h"
#include <iostream>

Database::Database(const std::string& filename) 
    : storage(std::make_unique<DiskStorage>(filename))/*, engine(schema, *storage)*/ {}
bool Database::hasDatabaseSelected() const{return !current_db.empty();}
const std::string& Database::currentDatabase() const{return current_db;}
void Database::execute(const std::string& query) {
    try {
        auto stmt = parseQuery(query);
        
        SematicAnalyzer analyzer(*this,*storage);//had schema instead
        analyzer.analyze(stmt);
	ExecutionEngine engine(*this,*storage);
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
    std::cout<<"ELVIS QUERY LANGUAGE:"
	    <<"\n\tVERSION 0.1:";
    std::cout << "\nESQL> ";
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
void Database::ensureDatabaseSelected() const{
	if(!hasDatabaseSelected()){
		throw std::runtime_error("No database selected. Use CREATE DATABASE or CREATE DATABASE first");
	}
}
