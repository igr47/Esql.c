#ifndef TERMINAL_INPUT_H
#define TERMINAL_INPUT_H

#include "shell_types.h"
#include <termios.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#endif

namespace esql {

class TerminalInput {
public:
    TerminalInput();
    ~TerminalInput();
    
    // Terminal control
    bool enable_raw_mode();
    void disable_raw_mode();
    void get_terminal_size(int& width, int& height);
    void clear_screen();
    void move_cursor(int row, int col);
    void erase_line();
    void erase_display_below();
    
    // Input handling
    KeyCode read_key();
    bool has_input_available();
    
    // Platform detection
    enum class Platform { Linux, Windows, Termux, Unknown };
    Platform detect_platform() const;
    bool is_termux() const { return current_platform_ == Platform::Termux; }
    
private:
    Platform current_platform_;
    
#ifdef _WIN32
    HANDLE hInput_;
    HANDLE hOutput_;
    DWORD original_mode_;
#else
    struct termios orig_termios_;

#endif
    
    // Platform-specific implementations
    KeyCode read_key_unix();
    //KeyCode read_key_windows();
    bool has_input_available_unix();
    //bool has_input_available_windows();
    bool enable_raw_mode_unix();
    //bool enable_raw_mode_windows();
    void disable_raw_mode_unix();
    //void disable_raw_mode_windows();
    void get_terminal_size_unix(int& width, int& height);
    //void get_terminal_size_windows(int& width, int& height);
    
    // Escape sequence parsing
    KeyCode parse_escape_sequence();
};

} // namespace esql

#endif
