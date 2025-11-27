#include "database.h"
#include "shell_includes/modern_shell.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cctype>
#include <thread>

Database::Database(const std::string& filename)
    : storage(std::make_unique<fractal::DiskStorage>(filename)), schema() {
    try {
        std::cout << "=== DATABASE INITIALIZATION ===" << std::endl;

        // Give diskstorage a moment to fully initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Get available databases
        auto databases = storage->listDatabases();
        std::cout << "Available databases: " << databases.size() << std::endl;
        
        if (databases.empty()) {
            std::cout << "No databases found, checking if we can create 'default'..." << std::endl;
            // Try to create default, but handle the case where file might exist
            try {
                storage->createDatabase("default");
                storage->useDatabase("default");
                setCurrentDatabase("default");
                std::cout << "Created new default database" << std::endl;
            } catch (const std::runtime_error& e) {
                // If creation fails because file exists, try to use it
                if (std::string(e.what()).find("already exists") != std::string::npos) {
                    std::cout << "Default database file exists, trying to use it..." << std::endl;
                    try {
                        storage->useDatabase("default");
                        setCurrentDatabase("default");
                        std::cout << "Successfully used existing default database" << std::endl;
                    } catch (const std::exception& use_error) {
                        std::cerr << "Failed to use existing default database: " << use_error.what() << std::endl;
                        throw;
                    }
                } else {
                    throw; // Re-throw other errors
                }
            }
        } else {
            std::cout << "Using existing database: " << databases[0] << std::endl;
            storage->useDatabase(databases[0]);
            setCurrentDatabase(databases[0]);
        }
        
        std::cout << "Current database: " << currentDatabase() << std::endl;
        std::cout << "=== INITIALIZATION COMPLETE ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing database: " << e.what() << "\n";
        throw;
    }
}
/*Database::Database(const std::string& filename)
    : storage(std::make_unique<fractal::DiskStorage>(filename)), schema() {
    try {
        std::cout << "=== DATABASE INITIALIZATION ===" << std::endl;

        // Load existing databases immediately
        storage->loadExistingDatabases();
        
        auto databases = storage->listDatabases();
        std::cout << "Available databases: " << databases.size() << std::endl;
        
        if (databases.empty()) {
            std::cout << "No databases found, creating 'default'..." << std::endl;
            storage->createDatabase("default");
            storage->useDatabase("default");
            setCurrentDatabase("default");
            std::cout << "Created new default database" << std::endl;
        } else {
            // Use the first available database
            std::cout << "Using existing database: " << databases[0] << std::endl;
            storage->useDatabase(databases[0]);
            setCurrentDatabase(databases[0]);
        }
        
        std::cout << "Current database: " << currentDatabase() << std::endl;
        std::cout << "=== INITIALIZATION COMPLETE ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing database: " << e.what() << "\n";
        throw;
    }
}*/

Database::~Database() {
	shutdown();
}


void Database::shutdown(){
	try{
		if(storage) {
            storage->flushPendingChanges();
			storage->checkpoint();
		}
	}catch (const std::exception& e) {
			std::cerr<< "Error during database shutdown: " <<e.what() <<std::endl;
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
    ModernShell shell(*this);
    if(hasDatabaseSelected()){
        shell.set_current_database(currentDatabase());
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
