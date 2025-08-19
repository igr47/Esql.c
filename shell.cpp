#include "shell.h"
#include "database.h"
#include <iostream>
#include <iomanip>
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

ESQLShell::ESQLShell(Database& db) : db(db), current_db("default") {
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
	    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios
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
	ioctl(STDOUT_FILENO , TIOCGWINS, &ws);
	screen_rows=ws.ws_row;
	creen_cols=ws.ws_col;
}

void ESQLShell::move_cursor_left(){
	if(cursor_pod>0){
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
	if(history_pos < (int)history.size()-1){
		history_pos++;
		refresh_line();
	}else if(history_po s== (int)history.size()-1){
		history_pos++;
		current_line.clear();
		cursor_pos=0;
		refresh_line();
	}
}

void ESQLShell::refresh_line(){
	if(history_pos>0 && history_pos<(int)history.size()){
		current_line=history[history_pos];
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
		query_stack_2.pop();
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
		current_line.insert(corsor_pos,1,c);
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

void ESQLShell::print_prompt() {
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::localtime(&now);

    if (use_colors) {
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
const std::unordered_set<std::string> keywords={
	"SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES",
	"UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DATABASE", "DATABASES",
	"DROP", "DELETE", "ALTER", "ADD", "RENAME", "USE", "SHOW", "DESCRIBE", "CLEAR"
};
const std::unordered_set<std::string> types={
	"BOOL", "BOOLEAN", "INT", "INTEGER", "FLOAT", "TEXT", "STRING"
};
const std::unordered_set<std::string> conditionals={
	"AND", "OR", "NOT", "NULL"
};
std::string ESQLShell::colorize_sql() {
	std::string coloured_line;
	bool in_double_quotes=false;
	bool in_single_quotes=false;
	std::string current_word;

	for(size_t i=0; i<current_line.length(); i++){
		c=current_line[i];

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
			//check if current word id a keyword
			std::string upper_word=current_word;
			std::transfer(upper_word.begin(),upper_word.end(),upper_word.begin(),::toupper);

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
	if(in_double_qoutes || in_single_quotes){
		coloured_line+=RESET;
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

void ESQLShell::print_result_table(const std::vector<std::vector<std::string>>& rows,
                                  const std::vector<std::string>& headers,
                                  long long duration_ms) {
    if (headers.empty()) return;

    std::vector<size_t> col_widths(headers.size());
    for (size_t i = 0; i < headers.size(); i++) {
        col_widths[i] = headers[i].length();
    }

    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size() && i < headers.size(); i++) {
            if (row[i].length() > col_widths[i]) {
                col_widths[i] = row[i].length();
            }
        }
    }

    if (use_colors) std::cout << CYAN;
    for (size_t i = 0; i < headers.size(); i++) {
        std::cout << std::left << std::setw(col_widths[i] + 2) << headers[i];
    }
    if (use_colors) std::cout << RESET;
    std::cout << "\n";

    for (size_t i = 0; i < headers.size(); i++) {
        std::cout << std::string(col_widths[i] + 2, '-');
    }
    std::cout << "\n";

    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size() && i < headers.size(); i++) {
            std::cout << std::left << std::setw(col_widths[i] + 2) << row[i];
        }
        std::cout << "\n";
    }

    if (duration_ms > 0) {
        std::cout << (use_colors ? GRAY : "") << "\nQuery executed in " << duration_ms << " ms"
                  << (use_colors ? RESET : "") << "\n";
    }
}



void ESQLShell::execute_command(const std::string& raw_cmd) {
    static std::string accumulated_cmd;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string cmd = expand_alias(raw_cmd);
    if (cmd.back() == '\\') {
        accumulated_cmd += cmd.substr(0, cmd.length() - 1) + " ";
        std::cout << (use_colors ? GRAY : "") << "Continuing command..." << (use_colors ? RESET : "") << "\n";
        return;
    } else {
        accumulated_cmd += cmd;
    }

    std::string upper_cmd = accumulated_cmd;
    std::transform(upper_cmd.begin(), upper_cmd.end(), upper_cmd.begin(), ::toupper);

    try {
        if (upper_cmd == "HELP") {
            show_help();
        } else if (upper_cmd == "CLEAR") {
            std::cout << "\033[2J\033[1;1H";
            print_banner();
        } else if (upper_cmd.find("USE DATABASE") == 0) {
            db.execute(accumulated_cmd);
            size_t pos = accumulated_cmd.find_last_of(' ');
            if (pos != std::string::npos && pos + 1 < accumulated_cmd.length()) {
                setCurrentDatabase(accumulated_cmd.substr(pos + 1));
                std::cout << (use_colors ? GREEN : "") << "Switched to database '" << current_db
                          << (use_colors ? RESET : "") << "\n";
            } else {
                std::cerr << (use_colors ? RED : "") << "Error: Invalid database name"
                          << (use_colors ? RESET : "") << "\n";
            }
        } else {
            // Just execute the command - Database::execute() handles output internally
            auto start = std::chrono::high_resolution_clock::now();
            db.execute(accumulated_cmd);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            // Only print timing info since the database handles the actual output
            std::cout << (use_colors ? GRAY : "") << "\nQuery executed in " << duration.count() << " ms"
                      << (use_colors ? RESET : "") << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << (use_colors ? RED : "") << "Error: " << e.what()
                  << (use_colors ? RESET : "") << "\n";
    }

    accumulated_cmd.clear();
}

void ESQLShell::handle_special_keys(){
	char seq[3];
	if(read(STDIN_FILENO, &seq[0], 1) != 1)return;
	if(read(STDIN_FILENO, &seq[1], 1) !=1)return;

	if(seq[0]='['){
		switch(seq[1]){
			case 'A': move_cursor_up(); break;
			case 'B': move_cursor_dowm(); break;
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

/*void ESQLShell::redraw_line() {
    std::cout << "\r\033[2K";
    print_prompt();

    std::string colored_input = colorize_sql(current_input);
    std::cout << colored_input;

    int term_width = 80;
    #ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        term_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
    #else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1) {
        term_width = ws.ws_col;
    }
    #endif

    size_t prompt_length = current_db.length() + 10;
    size_t visible_pos = calculate_visible_position(colored_input, cursor_pos);

    std::cout << "\r";
    print_prompt();
    if (visible_pos > 0) {
        std::string visible_part = colored_input;
        if (visible_part.length() > static_cast<size_t>(term_width - prompt_length)) {
            visible_part = visible_part.substr(0, term_width - prompt_length);
        }
        std::cout << visible_part.substr(0, calculate_visible_position(colored_input, cursor_pos));
    }

    std::cout.flush();
}*/

void ESQLShell::redraw_line() {
    // clear current line
    std::cout << "\r\033[2K";
    // prompt
    print_prompt();

    // Terminal width
    int term_width = 80;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        term_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1) term_width = ws.ws_col;
#endif

    // Colorize but compute visible window (no duplicate prints)
    std::string colored = colorize_sql(current_input);

    // Rough prompt length (no ANSI in prompt)
    size_t prompt_len = current_db.length() + 10;

    // Compute the visible window around the cursor
    size_t max_visible = (term_width > (int)prompt_len) ? (term_width - prompt_len) : 20;

    // Map logical cursor to visible column (skip ANSI)
    size_t cursor_col = calculate_visible_position(colored, cursor_pos);

    // Determine left offset so cursor stays in window
    size_t left = 0;
    if (cursor_col >= max_visible) left = cursor_col - max_visible + 1;

    // Build visible slice (skip ANSI safely by walking)
    std::string out;
    size_t vis = 0;
    bool in_esc = false;
    for (size_t i = 0; i < colored.size(); ++i) {
        char ch = colored[i];
        if (ch == '\033') in_esc = true;
        if (!in_esc) {
            if (vis >= left && vis < left + max_visible) {
                out.push_back(ch);
            }
            ++vis;
        } else {
            out.push_back(ch);
            if (ch == 'm') in_esc = false;
        }
    }

    std::cout << out;

    // Move cursor to correct column: prompt_len + (cursor_col - left)
    size_t target = prompt_len + (cursor_col >= left ? (cursor_col - left) : 0);
    std::cout << "\r\033[" << (target + 1) << "G"; // 1-based column
    std::cout.flush();
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

void ESQLShell::handle_unix_input() {
    #ifndef _WIN32
    int c;
    while ((c = getchar()) != EOF) {
        handle_special_keys(c);
        if (c == '\n') break;
    }
    #endif
}

void ESQLShell::run_interactive() {
    print_banner();
    enable_raw_mode();

    while (true) {
        print_prompt();
        std::cout << colorize_sql(current_input);
        std::cout.flush();

        if (current_platform == Platform::Windows) {
            handle_windows_input();
        } else {
            handle_unix_input();
        }

        std::string upper_input = current_input;
        std::transform(upper_input.begin(), upper_input.end(), upper_input.begin(), ::toupper);
        if (upper_input == "EXIT" || upper_input == "QUIT") {
            break;
        }
    }

    disable_raw_mode();
}

void ESQLShell::run() {
    run_interactive();
}

void ESQLShell::setCurrentDatabase(const std::string& db_name) {
    current_db = db_name;
    connection_status = "connected";
}

void ESQLShell::setConnectionStatus(const std::string& status) {
    connection_status = status;
}
