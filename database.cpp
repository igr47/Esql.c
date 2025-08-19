#include "database.h"
#include "shell.h"
#include <iostream>
#include <cctype>

Database::Database(const std::string& filename)
    : storage(std::make_unique<DiskStorage>(filename)), schema() {
    try {
        // Check if any databases exist
        auto databases = storage->listDatabases();
        if (databases.empty()) {
            // Create a default database if none exist
            storage->createDatabase("default");
            setCurrentDatabase("default");
        } else {
            // Select the first available database
            setCurrentDatabase(databases[0]);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error initializing database: " << e.what() << "\n";
        throw; // Re-throw to prevent partially initialized state
    }
}

bool Database::hasDatabaseSelected() const {
    return !current_db.empty();
}

const std::string& Database::currentDatabase() const {
    return current_db;
}

void Database::setCurrentDatabase(const std::string& dbName) {
    current_db = dbName;
}

void Database::execute(const std::string& query) {
    try {
        auto stmt = parseQuery(query);
        SematicAnalyzer analyzer(*this, *storage);
        analyzer.analyze(stmt); // Analyze once
        ExecutionEngine engine(*this, *storage);
        auto result = engine.execute(std::move(stmt));

        // Print results
        if (!result.columns.empty()) {
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
        } else {
            std::cout << "Query executed successfully.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        if (std::string(e.what()) == "No database selected. Use CREATE DATABASE or USE DATABASE first") {
            std::cerr << "Hint: Use 'CREATE DATABASE <name>;' or 'USE <name>;' to select a database.\n";
        }
    }
}
/*Database::QueryResult Database::execute(const std::string& query) {
    QueryResult result;
    try {
        auto stmt = parseQuery(query);
        SematicAnalyzer analyzer(*this, *storage);
        analyzer.analyze(stmt);
        ExecutionEngine engine(*this, *storage);
        result = engine.execute(std::move(stmt));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        if (std::string(e.what()) == "No database selected. Use CREATE DATABASE or USE DATABASE first") {
            std::cerr << "Hint: Use 'CREATE DATABASE <name>;' or 'USE <name>;' to select a database.\n";
        }
        // Return empty result on error
    }
    return result;
}*/

void Database::startInteractive() {
    // Reset terminal state
    /*std::cout << std::unitbuf;  // Enable automatic flushing
    std::ios_base::sync_with_stdio(true);
    
    std::cout << "\nELVIS QUERY LANGUAGE - Version 0.1\n";
    std::cout << "Type 'exit' or 'quit' to exit\n";
    std::cout << "Current database: " << (current_db.empty() ? "none" : current_db) << "\n\n";

    std::string input;
    while (true) {
        std::cout << "ESQL> " << std::flush;  // Explicit flush
        
        if (!std::getline(std::cin, input)) {
            if (std::cin.eof()) {
                std::cout << "\nExiting...\n";
                break;
            }
            std::cin.clear();
            continue;
        }

        // Trim input
        input.erase(0, input.find_first_not_of(" \t\n\r\f\v"));
        input.erase(input.find_last_not_of(" \t\n\r\f\v") + 1);

        if (input.empty()) continue;
        if (input == "exit" || input == "quit") break;

        try {
            execute(input);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }*/
	ESQLShell shell(*this);
	if(hasDatabaseSelected()){
		shell.setCurrentDatabase(currentDatabase());
	}
	shell.run();
    
}

std::unique_ptr<AST::Statement> Database::parseQuery(const std::string& query) {
    Lexer lexer(query);
    Parse parser(lexer); // Fixed typo: Parse -> Parser
    return parser.parse();
}

void Database::ensureDatabaseSelected() const {
    if (!hasDatabaseSelected()) {
        throw std::runtime_error("No database selected. Use CREATE DATABASE or USE DATABASE first");
    }
}
