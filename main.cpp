#include <iostream>
#include "parser.h"
#include "scanner.h"
#include <string>


int main() {
    std::string query;
    std::cout<<">> ";
    std::getline(std::cin,query);

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
