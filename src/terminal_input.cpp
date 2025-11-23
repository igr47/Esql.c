#include "terminal_input.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <conio.h>
#else
#include <fcntl.h>
#include <sys/select.h>
#endif

namespace esql {

TerminalInput::TerminalInput() {
    current_platform_ = detect_platform();
}

TerminalInput::~TerminalInput() {
    disable_raw_mode();
}

TerminalInput::Platform TerminalInput::detect_platform() const {
#ifdef _WIN32
    return Platform::Windows;
#else
    const char* term = getenv("TERM");
    const char* pkg = getenv("PREFIX");
    const char* android_root = getenv("ANDROID_ROOT");
    
    if ((pkg && std::string(pkg).find("com.termux") != std::string::npos) ||
        (android_root) ||
        access("/data/data/com.termux/files/usr/bin/login", F_OK) == 0 ||
        access("/data/data/com.termux/files/home", F_OK) == 0) {
        return Platform::Termux;
    }
    return Platform::Linux;
#endif
}

bool TerminalInput::enable_raw_mode() {
    switch (current_platform_) {
        case Platform::Windows:
            //return enable_raw_mode_windows();
            break;
        case Platform::Linux:
        case Platform::Termux:
            return enable_raw_mode_unix();
        default:
            return false;
    }
}

void TerminalInput::disable_raw_mode() {
    switch (current_platform_) {
        case Platform::Windows:
            //disable_raw_mode_windows();
            break;
        case Platform::Linux:
        case Platform::Termux:
            disable_raw_mode_unix();
            break;
        default:
            break;
    }
}

void TerminalInput::get_terminal_size(int& width, int& height) {
    switch (current_platform_) {
        case Platform::Windows:
            //get_terminal_size_windows(width, height);
            break;
        case Platform::Linux:
        case Platform::Termux:
            get_terminal_size_unix(width, height);
            break;
        default:
            width = 80;
            height = 24;
            break;
    }
    
    if (width < 40) width = 40;
    if (height < 10) height = 10;
}

KeyCode TerminalInput::read_key() {
    switch (current_platform_) {
        case Platform::Windows:
            //return read_key_windows();
            break;
        case Platform::Linux:
        case Platform::Termux:
            return read_key_unix();
        default:
            return KeyCode::None;
    }
}

bool TerminalInput::has_input_available() {
    switch (current_platform_) {
        case Platform::Windows:
            //return has_input_available_windows();
            break;
        case Platform::Linux:
        case Platform::Termux:
            return has_input_available_unix();
        default:
            return false;
    }
}

void TerminalInput::clear_screen() {
    std::cout << "\033[2J\033[H";
}

void TerminalInput::move_cursor(int row, int col) {
    std::cout << "\033[" << row << ";" << col << "H";
}

void TerminalInput::erase_line() {
    std::cout << "\033[K";
}

void TerminalInput::erase_display_below() {
    std::cout << "\033[J";
}

// ============ Unix/Linux/Termux Implementation ============

// ============ Windows Implementation ============



/*bool TerminalInput::enable_raw_mode_windows() {
    hInput_ = GetStdHandle(STD_INPUT_HANDLE);
    hOutput_ = GetStdHandle(STD_OUTPUT_HANDLE);

    if (hInput_ == INVALID_HANDLE_VALUE || hOutput_ == INVALID_HANDLE_VALUE) {
        return false;
    }

    if (!GetConsoleMode(hInput_, &original_mode_)) {
        return false;
    }

    DWORD new_mode = original_mode_;
    new_mode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT);
    new_mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

    return SetConsoleMode(hInput_, new_mode);
}

void TerminalInput::disable_raw_mode_windows() {
    if (hInput_ != INVALID_HANDLE_VALUE) {
        SetConsoleMode(hInput_, original_mode_);
    }
}

KeyCode TerminalInput::read_key_windows() {
    INPUT_RECORD input_record;
    DWORD events_read;

    while (true) {
        if (!ReadConsoleInput(hInput_, &input_record, 1, &events_read) || events_read == 0) {
            return KeyCode::None;
        }

        if (input_record.EventType == KEY_EVENT && input_record.Event.KeyEvent.bKeyDown) {
            auto key_event = input_record.Event.KeyEvent;
            WORD key_code = key_event.wVirtualKeyCode;

            // Handle special keys
            switch (key_code) {
                case VK_RETURN: return KeyCode::Enter;
                case VK_TAB: return KeyCode::Tab;
                case VK_BACK: return KeyCode::Backspace;
                case VK_DELETE: return KeyCode::Delete;
                case VK_UP: return KeyCode::Up;
                case VK_DOWN: return KeyCode::Down;
                case VK_LEFT: return KeyCode::Left;
                case VK_RIGHT: return KeyCode::Right;
                case VK_HOME: return KeyCode::Home;
                case VK_END: return KeyCode::End;
                case 'D':
                    if (key_event.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
                        return KeyCode::CtrlD;
                    break;
                case 'L':
                    if (key_event.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
                        return KeyCode::CtrlL;
                    break;
                case 'R':
                    if (key_event.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
                        return KeyCode::CtrlR;
                    break;
            }

            // Handle character input
            if (key_event.uChar.AsciiChar >= 32 && key_event.uChar.AsciiChar <= 126) {
                return KeyCode::Character;
            }
        }
    }

    return KeyCode::None;
}

bool TerminalInput::has_input_available_windows() {
    DWORD events_available;
    return GetNumberOfConsoleInputEvents(hInput_, &events_available) && events_available > 0;
}

void TerminalInput::get_terminal_size_windows(int& width, int& height) {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(hOutput_, &csbi)) {
        width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    } else {
        width = 80;
        height = 24;
    }
}*/



bool TerminalInput::enable_raw_mode_unix() {
    if (tcgetattr(STDIN_FILENO, &orig_termios_) == -1) {
        return false;
    }
    
    struct termios raw = orig_termios_;
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_iflag &= ~(IXON | ICRNL | INPCK | ISTRIP);
    raw.c_cflag |= (CS8);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    
    return tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != -1;
}

void TerminalInput::disable_raw_mode_unix() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios_);
}

KeyCode TerminalInput::read_key_unix() {
    char c;
    ssize_t n = read(STDIN_FILENO, &c, 1);
    
    if (n != 1) {
        return KeyCode::None;
    }
    
    if (c == '\033') {
        return parse_escape_sequence();
    }
    
    switch (static_cast<unsigned char>(c)) {
        case '\n': case '\r': return KeyCode::Enter;
        case '\t': return KeyCode::Tab;
        case 127: return KeyCode::Backspace;
        case 8: return KeyCode::Delete;
        case 4: return KeyCode::CtrlD;
        case 12: return KeyCode::CtrlL;
        case 18: return KeyCode::CtrlR;
        default:
            if (c >= 32 && c <= 126) {
                return KeyCode::Character;
            }
            return KeyCode::None;
    }
}

KeyCode TerminalInput::parse_escape_sequence() {
    char seq[3] = {0};
    
    struct timeval tv = {0, 100000}; // 100ms timeout
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0) {
        return KeyCode::Escape;
    }
    
    if (read(STDIN_FILENO, &seq[0], 1) != 1) {
        return KeyCode::Escape;
    }
    
    if (seq[0] == '[') {
        if (read(STDIN_FILENO, &seq[1], 1) != 1) {
            return KeyCode::Escape;
        }
        
        switch (seq[1]) {
            case 'A': return KeyCode::Up;
            case 'B': return KeyCode::Down;
            case 'C': return KeyCode::Right;
            case 'D': return KeyCode::Left;
            case 'H': return KeyCode::Home;
            case 'F': return KeyCode::End;
            case '1': 
                if (read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == '~') 
                    return KeyCode::Home;
                break;
            case '4': 
                if (read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == '~') 
                    return KeyCode::End;
                break;
        }
    }
    
    // Consume any remaining escape sequence characters
    while (has_input_available()) {
        char dummy;
        read(STDIN_FILENO, &dummy, 1);
    }
    
    return KeyCode::None;
}

bool TerminalInput::has_input_available_unix() {
    struct timeval tv = {0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    return select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0;
}

void TerminalInput::get_terminal_size_unix(int& width, int& height) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        width = ws.ws_col;
        height = ws.ws_row;
    } else {
        const char* columns = getenv("COLUMNS");
        const char* lines = getenv("LINES");
        width = columns ? std::atoi(columns) : 80;
        height = lines ? std::atoi(lines) : 24;
    }
}

}
 
