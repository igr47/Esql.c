#include "database.h"
#include <iostream>

int main() {
    Database db;
    
    // Example usage
    //db.execute("CREATE TABLE users(name:TEXT, age:INT)");
    //db.execute("INSERT INTO users VALUES ('Alice', 25)");
    //db.execute("INSERT INTO users VALUES ('Bob', 30)");
    //db.execute("SELECT name, age FROM users WHERE age > 20");
    
    // Start interactive shell
    db.startInteractive();
    
    return 0;
}
