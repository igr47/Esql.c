#include <iostream>
#include "database.h"

int main() {
    try {
        std::cout << "Starting database system...\n";
        Database db("mydb");
        std::cout << "Database initialized successfully\n";
        std::cout << "Starting interactive shell...\n";
        db.startInteractive();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
