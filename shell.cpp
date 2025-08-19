#include "shell.h"
#include "database.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <ctime>
#include <algorithm>
#include <regex>
#include <chrono>
#include <thread>
#include <cstring>
#include <sstream>



ESQLShell::Platform ESQLShell::detect_platform() {
    #ifdef _WIN32
    return Platform::Windows;
    #else
    const char* term = getenv("TERM");
    const char* pkg = getenv("PREFIX");

    if ((pkg && strstr(pkg, "com.termux")) ||
        access("/data/data/com.termux/files/usr", F_OK) == 0 ||
        (term && strstr(term, "termux"))) {
        return Platform::Termux;
    }
    if (term) {
        return Platform::Linux;
    }
    return Platform::Unknown;
    #endif
}

void ESQLShell::platform_specific_init() {
    current_platform = detect_platform();

    switch (current_platform) {
        case Platform::Termux:
            use_colors = true;
            break;
        case Platform::Windows:
            use_colors = setup_windows_terminal();
            break;
        case Platform::Linux:
            #ifndef _WIN32
            using_history();
            read_history(".esql_history");
            #endif
            use_colors = true;
            break;
        default:
            use_colors = false;
            break;
    }
}

ESQLShell::ESQLShell(Database& db) : db(db), current_db("default"), history_pos(0),cursor_pos(0) {
    enable_raw_mode();
    get_window_size();
    platform_specific_init();
}

ESQLShell::~ESQLShell() {
    disable_raw_mode();
    if (current_platform != Platform::Windows) {
        write_history(".esql_history");
    } else {
        disable_raw_mode();;
    }
}

bool ESQLShell::setup_windows_terminal() {
    #ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return false;

    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return false;

    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT;
    if (!SetConsoleMode(hOut, dwMode)) return false;

    SetConsoleOutputCP(CP_UTF8);
    return true;
    #else
    return false;
    #endif
}

void ESQLShell::restore_windows_terminal() {
    #ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return;

    dwMode &= ~(ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT);
    SetConsoleMode(hOut, dwMode);
    #endif
}

void ESQLShell::enable_raw_mode() {
    //if (raw_mode) return;

    switch (current_platform) {
        case Platform::Linux:
        case Platform::Termux: {
            //if (tcgetattr(STDIN_FILENO, &orig_termios) == -1) return;
	    tcgetattr(STDIN_FILENO, &orig_termios);
            struct termios raw = orig_termios;

            raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
            raw.c_oflag &= ~(OPOST);
            raw.c_cflag |= (CS8);
            raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
            raw.c_cc[VMIN] = 1;
            raw.c_cc[VTIME] = 0;

            //if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == -1) return;
	    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
            break;
        }
        case Platform::Windows:
            break;
        default:
            break;
    }

    //raw_mode = true;
}

void ESQLShell::disable_raw_mode() {
    //if (!raw_mode) return;

    switch (current_platform) {
        case Platform::Linux:
        case Platform::Termux:
            //if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios) == -1) return;
	    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
            //std::cout << "\033[?25h";
            //std::cout << "\033[?1049l";
            break;
        case Platform::Windows:
        default:
            break;
    }

    //raw_mode = false;
}

void ESQLShell::print_banner() {
    if (use_colors) {
        std::cout << BLUE << "\n";
        std::cout << "   _____ _      _       ____  ____  \n";
        std::cout << "  | ____| | ___| |_ ___| __ )| __ ) \n";
        std::cout << "  |  _| | |/ _ \\ __/ __|  _ \\|  _ \\ \n";
        std::cout << "  | |___| |  __/ |_\\__ \\ |_) | |_) |\n";
        std::cout << "  |_____|_|\\___|\\__|___/____/|____/ \n";
        std::cout << RESET << "  Database Management System v1.2\n\n";
    } else {
        std::cout << "\n";
        std::cout << "   ESQL Database Management System v1.2\n\n";
    }
    std::cout << "Type 'help' for commands, 'exit' to quit\n";
    update_connection_status();
}

void ESQLShell::update_connection_status() {
    std::cout << "Status: ";
    if (connection_status == "connected") {
        std::cout << (use_colors ? GREEN : "") << "● " << (use_colors ? RESET : "")
                  << "Connected to: " << (use_colors ? WHITE : "") << current_db
                  << (use_colors ? RESET : "");
    } else {
        std::cout << (use_colors ? RED : "") << "● " << (use_colors ? RESET : "")
                  << "Disconnected";
    }
    std::cout << "\n\n";
}

void ESQLShell::get_window_size(){
	struct winsize ws;
	ioctl(STDOUT_FILENO , TIOCGWINSZ, &ws);
	screen_rows=ws.ws_row;
	screen_cols=ws.ws_col;
}

void ESQLShell::move_cursor_left(){
	if(cursor_pos>0){
		cursor_pos--;
		std::cout<<"\033[D";
		std::cout.flush();
	}
}

void ESQLShell::move_cursor_right(){
	if(cursor_pos<current_line.length()){
		cursor_pos++;
		std::cout<<"\033[C";
		std::cout.flush();
	}
}

void ESQLShell::move_cursor_up(){
	if(history_pos>0){
		history_pos--;
		refresh_line();
	}
}

void ESQLShell::move_cursor_down(){
	if(history_pos < (int)command_history.size()-1){
		history_pos++;
		refresh_line();
	}else if(history_pos== (int)command_history.size()-1){
		history_pos++;
		current_line.clear();
		cursor_pos=0;
		refresh_line();
	}
}

void ESQLShell::refresh_line(){
	if(history_pos>0 && history_pos<(int)command_history.size()){
		current_line=command_history[history_pos];
	}
	cursor_pos=current_line.length();
	redraw_line();
}

void ESQLShell::previous_command_up(){
	if(!query_stack.empty()){
	        current_line=query_stack.top();
		query_stack.pop();
		cursor_pos=current_line.length();
		query_stack2.push(current_line);
		redraw_line();
	}else{
		std::cout<<"\a";//Beep
		std::cout.flush();
	}
}

void ESQLShell::previous_command_down(){
	if(!query_stack2.empty()){
		current_line=query_stack2.top();
		query_stack2.pop();
		cursor_pos=current_line.length();
		query_stack.push(current_line);
		redraw_line();
	}else{
		std::cout<<"\a";
		std::cout.flush();
	}
}

void ESQLShell::insert_char(char c){
	if(cursor_pos==current_line.length()){
		current_line+=c;
	}else{
		current_line.insert(cursor_pos,1,c);
	}
	cursor_pos++;
	redraw_line();
}

void ESQLShell::delete_char(){
	if(cursor_pos>0 && !current_line.empty()){
		current_line.erase(cursor_pos-1,1);
		cursor_pos--;
		redraw_line();
	}
}

void ESQLShell::redraw_line(){
	//move to start of lone and clear
	std::cout<<"\r\033[K";
	//Reprint prompt
	print_prompt();
	//Reprint colourizedline
	std::cout<<colorize_sql();
	//Position cursor correctly
	if(cursor_pos<current_line.length()){
		int move_back=current_line.length()-cursor_pos;
		std::cout<<"\033["<<move_back<<"D";
	}
	std::cout.flush();
}

void ESQLShell::clear_line(){
	std::cout<<"\r\033[K";
	std::cout.flush();
}

void ESQLShell::print_results(const ExecutionEngine::ResultSet& result,double duration){
	if(result.columns.empty()){
		std::cout<<GREEN<<"Querry executed successfully.\n";
		return;
	}

	//calculate column widths
	std::vector<size_t> column_widths(result.columns.size());
	for(size_t i=0; i<result.columns.size(); i++){
		column_widths[i]=result.columns[i].length();
		for(const auto& row : result.rows){
			if(i < row.size() && row[i].length() > column_widths[i]){
				column_widths[i] = row[i].length();
			}
		}
		column_widths[1]+=2;
	}
	//Print header 
	std::cout<<CYAN;
	for(size_t i=0; i<result.columns.size(); i++){
		std::cout<<std::left<<std::setw(column_widths[i]) <<result.columns[i];
	}
	std::cout<<RESET;
	//Print separator
	 for(size_t i=0; i<result.columns.size(); i++){
		 std::cout<<std::string(column_widths[i],'-')<<"";
	}
	 std::cout<<"\n";

	 //Print rows
	 for(const auto& row : result.rows){
		 for(size_t i=0; i<row.size(); ++i){
			 std::cout<<std::left<<std::setw(column_widths[i])<<row[i];
		}
		 std::cout<<"\n";
	}
	 //Print row count and timing
	 std::cout<<"(" <<result.rows.size() <<"row" <<(result.rows.size()!=1 ? "s" : " ")<<")\n";
	 std::cout<<GRAY <<"Time: "<<std::fixed<<std::setprecision(4)<<duration<< "seconds\n"<<RESET;
}

void ESQLShell::handle_enter(){
	std::cout<<"\r\n";
	if(!current_line.empty()){
		command_history.push_back(current_line);
		history_pos=command_history.size();
		query_stack.push(current_line);

		std::string upper_line=current_line;
		std::transform(upper_line.begin(),upper_line.end(),upper_line.begin(),::toupper);
		if(upper_line=="EXIT" || upper_line=="QUIT"){
			disable_raw_mode();
			exit(0);
		}else if(upper_line=="HELP"){
			show_help();
		}else if(upper_line=="CLEAR"){
			std::cout<<"\033[2j\033[H";
			print_banner();
			print_prompt();
			redraw_line();
		}
		auto start=std::chrono::high_resolution_clock::now();
		try{
			//auto start=std::chrono::high_resolution_clock::now();
			auto[result,duration]=db.executeQuery(current_line);
			auto end=std::chrono::high_resolution_clock::now();
			double shell_duration=std::chrono::duration<double>(end-start).count();
			print_results(result,shell_duration);
		}catch(const std::exception& e){
			auto end=std::chrono::high_resolution_clock::now();
			double duration=std::chrono::duration<double>(end-start).count();
			std::cerr<<RED<<"Error: "<<e.what()<<RESET<<"\n";
			std::cerr<<GRAY<<"Time: "<<std::fixed<<std::setprecision(4)<<duration<< "seconds.\n";
			
			if(std::string(e.what())=="No database selected.Use CREATE DATABASE or USE DATABASE first"){
				std::cerr<<YELLOW<<"Hint: Use 'CREATE DATABASE <name>;' or 'USE DATABASE <name>' to select database,\n"<<RESET;
			}
		}
	}
}
void ESQLShell::print_prompt() {
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::localtime(&now);

    if (use_colors){
        std::cout << YELLOW << "[" << std::setw(2) << std::setfill('0') << ltm->tm_hour << ":"
                  << std::setw(2) << std::setfill('0') << ltm->tm_min << "] " << RESET;
        std::cout << (connection_status == "connected" ? GREEN : RED) << "● " << RESET;
        std::cout << GRAY << current_db << RESET << "> ";
    } else {
        std::cout << "[" << std::setw(2) << std::setfill('0') << ltm->tm_hour << ":"
                  << std::setw(2) << std::setfill('0') << ltm->tm_min << "] ";
        std::cout << (connection_status == "connected" ? "*" : "!") << " ";
        std::cout << current_db << "> ";
    }
}
const std::unordered_set<std::string> ESQLShell::keywords={
	"SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES",
	"UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DATABASE", "DATABASES",
	"DROP", "DELETE", "ALTER", "ADD", "RENAME", "USE", "SHOW", "DESCRIBE", "CLEAR"
};
const std::unordered_set<std::string> ESQLShell::types={
	"BOOL", "BOOLEAN", "INT", "INTEGER", "FLOAT", "TEXT", "STRING"
};
const std::unordered_set<std::string> ESQLShell::conditionals={
	"AND", "OR", "NOT", "NULL"
};
std::string ESQLShell::colorize_sql() {
	std::string coloured_line;
	bool in_double_quotes=false;
	bool in_single_quotes=false;
	std::string current_word;

	for(size_t i=0; i<current_line.length(); i++){
		char c=current_line[i];

		if(c=='"' && !in_single_quotes){
			if(!in_double_quotes){
				coloured_line+=GRAY;
			}else{
				coloured_line+=RESET;
			}
			in_double_quotes=!in_double_quotes;
			coloured_line+=c;
			continue;
		}

		if(c=='\'' && !in_double_quotes){
			if(!in_single_quotes){
				coloured_line+=GRAY;
			}else{
				coloured_line+=RESET;
			}
			in_single_quotes=!in_single_quotes;
			coloured_line+=c;
			continue;
		}
		//Handle pancuation
		
		if(c=='*' || c=='(' || c==')' || c=='=' || c=='<' || c=='>'){
			if(!current_word.empty()){
			        //check if current word id a keyword
			        std::string upper_word=current_word;
			        std::transform(upper_word.begin(),upper_word.end(),upper_word.begin(),::toupper);

			        if(keywords.count(upper_word)){
				        coloured_line+=MAGENTA + current_word + RESET;
			        }else if(types.count(upper_word)){
				        coloured_line+=BLUE + current_word + RESET;
			        }else if(conditionals.count(upper_word)){
				        coloured_line+=CYAN + current_word +RESET;
			        }else{
				        coloured_line=current_word;
			        }
			        current_word.clear();
		        }
		        coloured_line+=CYAN;
		        coloured_line+=c;
		        coloured_line+=RESET;
		        continue;
	        }
	        //Handle word bounderies
	        if(isalnum(c)){
		        current_word+=c;
	        }else{
		        if(!current_word.empty()){
			        //chek if word is keyword
			        std::string upper_word=current_word;
			        std::transform(upper_word.begin(),upper_word.end(),upper_word.begin(),::toupper);

			        if(keywords.count(upper_word)){
				        coloured_line+=MAGENTA + current_word + RESET;
			        }else if(types.count(upper_word)){
				        coloured_line+= BLUE + current_word + RESET;
			        }else if(conditionals.count(upper_word)){
				        coloured_line+=CYAN +current_word +RESET;
			        }else{
				        coloured_line+=current_word;
			        }
			        current_word.clear();
		        }
	        }
	        if(!current_word.empty()){
		        std::string upper_word=current_word;                     
		        std::transform(upper_word.begin(),upper_word.end(),upper_word.begin(),::toupper);           
		        if(keywords.count(upper_word)){                                                          
			        coloured_line+=MAGENTA + current_word + RESET;                                     
		        }else if(types.count(upper_word)){                                                         
			        coloured_line+= BLUE + current_word + RESET;                                       
		        }else if(conditionals.count(upper_word)){                                                   
			        coloured_line+=CYAN +current_word +RESET;                                           
		        }else{                                                                                      
			        coloured_line+=current_word;                                                       
		        }                                                                                
	        }                                                                                               
	        if(in_double_quotes || in_single_quotes){
		        coloured_line+=RESET;
	        }
	}
	return coloured_line;
    
}

void ESQLShell::show_help() {
    std::cout << "\n" << (use_colors ? CYAN : "") << "Available commands:" << (use_colors ? RESET : "") << "\n";
    std::cout << "  " << (use_colors ? MAGENTA : "") << "SELECT" << (use_colors ? WHITE : "") << "    - Query data from tables\n" <<RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "INSERT" << (use_colors ? WHITE : "") << "    - Add new records\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "UPDATE" << (use_colors ? WHITE : "") << "    - Modify existing records\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "DELETE" << (use_colors ? WHITE : "") << "    - Remove records\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "CREATE" << (use_colors ? WHITE : "") << "    - Create new tables/databases\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "DROP" << (use_colors ? WHITE : "") << "      - Remove tables/databases\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "ALTER" << (use_colors ? WHITE : "") << "     - Modify table structure\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "USE" << (use_colors ? WHITE : "") << "       - Switch databases\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "SHOW" << (use_colors ? WHITE : "") << "      - Display database info\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "HELP" << (use_colors ? WHITE : "") << "      - Show this help\n" << RESET;
    std::cout << "  " << (use_colors ? MAGENTA : "") << "EXIT" << (use_colors ? WHITE : "") << "      - Quit the shell\n" << RESET;
    std::cout << "\n";
}

std::string ESQLShell::expand_alias(const std::string& input) {
    static const std::map<std::string, std::string> aliases = {
        {"ls", "SHOW TABLES"},
        {"desc", "DESCRIBE"},
        {"cls", "CLEAR"}
    };

    std::istringstream iss(input);
    std::string first_word;
    iss >> first_word;

    auto it = aliases.find(first_word);
    if (it != aliases.end()) {
        return it->second + input.substr(first_word.length());
    }
    return input;
}


void ESQLShell::handle_special_keys(){
	char seq[3];
	if(read(STDIN_FILENO, &seq[0], 1) != 1)return;
	if(read(STDIN_FILENO, &seq[1], 1) !=1)return;

	if(seq[0]=='['){
		switch(seq[1]){
			case 'A': move_cursor_up(); break;
			case 'B': move_cursor_down(); break;
			case 'C': move_cursor_right(); break;
			case 'D': move_cursor_left(); break;
			case '5':
				  if(read(STDIN_FILENO, &seq[2], 1) ==1 && seq[2]=='~'){
					  previous_command_up();
				  }
			case '6':
				  if(read(STDIN_FILENO, &seq[2], 1) ==1 && seq[2]=='~'){
					  previous_command_down();
				  }
			case 'H':
				  cursor_pos=0;
				  redraw_line();
				  break;
			case 'F': 
				  cursor_pos=current_line.length();
				  redraw_line();
				  break;
		}
	}
}


size_t ESQLShell::calculate_visible_position(const std::string& str, size_t logical_pos) {
    size_t visible_pos = 0;
    bool in_escape = false;

    for (size_t i = 0; i < str.length() && i < logical_pos; ++i) {
        if (str[i] == '\033') {
            in_escape = true;
        } else if (in_escape) {
            if (str[i] == 'm' || str[i] == 'H' || str[i] == 'K') {
                in_escape = false;
            }
        } else {
            visible_pos++;
        }
    }

    return visible_pos;
}




void ESQLShell::handle_windows_input() {
    #ifdef _WIN32
    while (true) {
        int c = _getch();
        if (c == 0 || c == 224) {
            c = _getch();
            switch (c) {
                case 72:
                    if (!command_history.empty()) {
                        current_input = command_history.back();
                        cursor_pos = current_input.length();
                    }
                    break;
                case 80:
                    current_input.clear();
                    cursor_pos = 0;
                    break;
                case 75:
                    if (cursor_pos > 0) cursor_pos--;
                    break;
                case 77:
                    if (cursor_pos < current_input.length()) cursor_pos++;
                    break;
            }
            redraw_line();
        } else {
            handle_special_keys(c);
            if (c == 13) break;
        }
    }
    #endif	
}


size_t ESQLShell::calculate_visible_length(const std::string& str){
	size_t length = 0;
        bool in_escape = false;

        for (char c : str) {
            if (c == '\033') { // Start of ANSI escape sequence
                in_escape = true;
                continue;
            }

            if (in_escape) {
                if (c == 'm') { // End of ANSI escape sequence
                    in_escape = false;
                }
                continue;
            }

            length++; // Count only non-escape characters
        }

        return length;
}


std::string ESQLShell::get_prompt_string() {
	std::stringstream prompt_ss;
        std::time_t now = std::time(nullptr);
        std::tm* ltm = std::localtime(&now);

        if (use_colors) {
            prompt_ss << YELLOW << "[" << std::setw(2) << std::setfill('0') << ltm->tm_hour << ":"
                     << std::setw(2) << std::setfill('0') << ltm->tm_min << "] " << RESET;
            prompt_ss << (connection_status == "connected" ? GREEN : RED) << "● " << RESET;
            prompt_ss << GRAY << current_db << RESET << "> ";
        } else {
            prompt_ss << "[" << std::setw(2) << std::setfill('0') << ltm->tm_hour << ":"
                     << std::setw(2) << std::setfill('0') << ltm->tm_min << "] ";
            prompt_ss << (connection_status == "connected" ? "*" : "!") << " ";
            prompt_ss << current_db << "> ";
        }

        return prompt_ss.str();
}

void ESQLShell::update_prompt_cache() {
	std::string current_prompt = get_prompt_string();
        if (current_prompt != last_prompt) {
            last_prompt = current_prompt;
            cached_prompt_length = calculate_visible_length(current_prompt);
        }
}

void ESQLShell::handle_unix_input() {
    #ifndef _WIN32
    int c;
    while ((c = getchar()) != EOF) {
        //handle_special_keys(c);
        if (c == '\n') break;
    }
    #endif
}


void ESQLShell::run() {
	print_banner();

	while(true){
		char c;

		if(read(STDIN_FILENO, &c ,1) !=1)continue;

		if(c=='\x1b'){//Escape sequence
			handle_special_keys();
		}else if(c=='\r' || c=='\n'){
			handle_enter();
		}else if(c==127 || c=='\b'){
			delete_char();
		}else if(c=='\t'){

		}else if(isprint(c)){
			insert_char(c);

			//Handle line warping
			update_prompt_cache();
			int line_length = cached_prompt_length + current_line.length();
			if (line_length >= screen_cols){
				std::cout << "\n";
			}
		}
	}
}

void ESQLShell::setCurrentDatabase(const std::string& db_name) {
    current_db = db_name;
    connection_status = "connected";
}

void ESQLShell::setConnectionStatus(const std::string& status) {
    connection_status = status;
}
