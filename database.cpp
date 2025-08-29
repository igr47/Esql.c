#include "database.h"
#include "shell.h"
#include <iostream>
#include <chrono>
#include <iomanip>
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

// In database.cpp - Database::executeQuery() method
std::pair<ExecutionEngine::ResultSet, double> Database::executeQuery(const std::string& query){
    auto start=std::chrono::high_resolution_clock::now();
    try{
        auto stmt = parseQuery(query);
        
        // Handle USE DATABASE statements directly
        if (auto useStmt = dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())) {
            storage->useDatabase(useStmt->dbName);
            setCurrentDatabase(useStmt->dbName);
            
            ExecutionEngine::ResultSet result;
            result.columns = {"Status"};
            result.rows = {{"Database changed to '" + useStmt->dbName + "'"}};
            
            auto end=std::chrono::high_resolution_clock::now();
            double duration=std::chrono::duration<double>(end-start).count();
            return {result, duration};
        }
        
        // For other statements, ensure a database is selected
        ensureDatabaseSelected();
        
        SematicAnalyzer analyzer(*this, *storage);                                                         
        analyzer.analyze(stmt);                                                        
        ExecutionEngine engine(*this, *storage);                                                           
        auto result = engine.execute(std::move(stmt));
        
        auto end=std::chrono::high_resolution_clock::now();
        double duration=std::chrono::duration<double>(end-start).count();
        return {result,duration};
    }catch(const std::exception& e){
        auto end=std::chrono::high_resolution_clock::now();
        double duration=std::chrono::duration<double>(end-start).count();
        throw;
    }
}

/*std::pair<ExecutionEngine::ResultSet, double> Database::executeQuery(const std::string& query){
    auto start=std::chrono::high_resolution_clock::now();
    try{
        auto stmt = parseQuery(query);

        // Check if this is a USE DATABASE statement
        if (auto useStmt = dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())) {
            setCurrentDatabase(useStmt->dbName);
            storage->useDatabase(useStmt->dbName);
            ExecutionEngine::ResultSet result;
            result.columns = {"Status"};
            result.rows = {{"Switched to database: " + useStmt->dbName}};

            auto end=std::chrono::high_resolution_clock::now();
            double duration=std::chrono::duration<double>(end-start).count();
            return {result, duration};
        }

        // For other statements, ensure a database is selected
        ensureDatabaseSelected();

        SematicAnalyzer analyzer(*this, *storage);
        analyzer.analyze(stmt); // Analyze once
        ExecutionEngine engine(*this, *storage);
        auto result = engine.execute(std::move(stmt));

        auto end=std::chrono::high_resolution_clock::now();
        double duration=std::chrono::duration<double>(end-start).count();
        return {result,duration};
    }catch(const std::exception& e){
        auto end=std::chrono::high_resolution_clock::now();
        double duration=std::chrono::duration<double>(end-start).count();
        throw;
    }
}*/

void Database::execute(const std::string& query) {
    try {
        auto [result,duration]=executeQuery(query);

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
            std::cout<<"Time: "<<std::fixed<<std::setprecision(4)<<duration <<" seconds\n";
        } else {
            std::cout << "Query executed successfully.\n";
            std::cout<<"Time: "<<std::fixed<<std::setprecision(4)<< duration <<" seconds\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        if (std::string(e.what()) == "No database selected. Use CREATE DATABASE or USE DATABASE first") {
            std::cerr << "Hint: Use 'CREATE DATABASE <name>;' or 'USE <name>;' to select a database.\n";
        }
    }
}

void Database::startInteractive() {
    ESQLShell shell(*this);
    if(hasDatabaseSelected()){
        shell.setCurrentDatabase(currentDatabase());
    }
    shell.run();
}

std::unique_ptr<AST::Statement> Database::parseQuery(const std::string& query) {
    Lexer lexer(query);
    Parse parser(lexer);
    return parser.parse();
}

void Database::ensureDatabaseSelected() const {
    if (!hasDatabaseSelected()) {
        throw std::runtime_error("No database selected. Use CREATE DATABASE or USE DATABASE first");
    }
}
