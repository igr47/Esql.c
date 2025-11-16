#include <iostream>
//#include <csignal>
#include "database.h"

//Database* global =nullptr;

/*void signalHandler(int signal) {
	std::cout <<"\nReceived signal " << signal << ", shutting down gracefull...." <<std::endl;
	if(globalDb) {
		delete globalDb;
		globalDb = nullptr;
	}

	exit(signal);
}*/

int main() {
    try {
        //std::cout << Starting database system...\n;

	//std::signal(SIGINT,signalHandler);
	//std::signal(SIGINT,signalHandler);
        Database db("./databases/");
        std::cout << "Database initialized successfully\n";
        std::cout << "Starting interactive shell...\n";
        db.startInteractive();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
