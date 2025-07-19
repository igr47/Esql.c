#include <iostream>
#include "parser.h"
#include "scanner.h"


int main() {
    std::string query = "SELECT id, name, age FROM users WHERE age > 18 AND status = 'active'";

    try {
        Lexer lexer(query);
        Parse parser(lexer);
        auto selectStmt = parser.parse();

        // At this point you have the parsed AST
        std::cout << "Query parsed successfully!" << std::endl;

        // You would typically pass the AST to an executor or analyzer here

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
