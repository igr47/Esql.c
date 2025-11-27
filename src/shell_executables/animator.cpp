#include "shell_includes/animator.h"

std::string ConsoleAnimator::compressTextToFit(const std::string& text, int max_width) {
        std::string compressed = text;

        // First, try with normal spacing
        if (compressed.length() <= max_width) {
            return compressed;
        }

        // Remove spaces between words to make it fit
        std::string no_spaces;
        for (char c : compressed) {
            if (c != ' ') {
                no_spaces += c;
            }
        }

        if (no_spaces.length() <= max_width) {
                        return no_spaces;
        }

        // If still too long, truncate with ellipsis
        if (max_width > 3) {
            return no_spaces.substr(0, max_width - 3) + "...";
        }

        return no_spaces.substr(0, max_width);
}


void ConsoleAnimator::printEnlargedChar(char c, bool isLast) {
    //const std::string bold = "\033[1m";
    std::string green = "\033[32m";
    const std::string normal = "\033[0m";
    std::cout << green << c << normal;
    if (!isLast) {
        std::cout << " ";
    }
}

void ConsoleAnimator::printNormalChar(char c, bool isLast) {
    const std::string green = "\033[32m";
    const std::string normal = "\033[0m";
    std::cout << green << normal << c;
    if (!isLast) {
        std::cout << " ";
    }
}

void ConsoleAnimator::animateText(const std::string& text, int durationMs) {
           hideCursor();

        // Calculate available width (leave space for spinner and prefix)
        int available_width = terminal_width_ - 15; // Account for "[+] " prefix and spinner
        if (available_width < 20) available_width = 20; // Minimum reasonable width

        // Compress text to fit on one line
        std::string display_text = compressTextToFit(text, available_width);
        int totalChars = display_text.length();

        int frameDelay = 100;
        int totalFrames = durationMs / frameDelay;

        int charsPerFrame = std::max(1, totalFrames / totalChars);

        // Initial display
        std::cout << "\033[32m"; // Set green color
        std::cout << "\r";
        for (int i = 0; i < totalChars; i++) {
            //printNormalChar(display_text[i]);
            std::cout << display_text[i];
        }
        std::cout.flush();

        // Animation loop
        for (int frame = 0; frame < totalFrames; frame++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameDelay));
            std::cout << "\r";
            std::cout << "\033[32m"; // Ensure green color

            // Animate the text
            for (int i = 0; i < totalChars; i++) {
                int charFrame = (frame - i * charsPerFrame) % (charsPerFrame * 3);

                if (charFrame >= 0 && charFrame < charsPerFrame) {
                    //printEnlargedChar(display_text[i]);
                    std::cout << "\033[1;32m" << display_text[i] << "\033[0;32m";
                } else {
                    //printNormalChar(display_text[i]);
                     std::cout << display_text[i];
                }
            }

            // Add rotating spinner
            std::string spinner = "|/-\\";
            std::cout << " " << spinner[frame % spinner.length()];
                        std::cout.flush();
        }

        // Final state - all characters normal
        std::cout << "\r";
        std::cout << "\033[32m"; 
        for (int i = 0; i < totalChars; i++) {
            //printNormalChar(display_text[i]);
             std::cout << display_text[i];
        }
        //std::cout << " |" << std::endl;
        std::cout << " |\033[0m" << std::endl; // Reset color

        showCursor();
}

std::string WaveAnimator::compressTextToFit(const std::string& text, int max_width) {
        std::string compressed = text;

        // First, try with normal spacing
        if (compressed.length() <= max_width) {
            return compressed;
        }

        // Remove spaces between words to make it fit
        std::string no_spaces;
        for (char c : compressed) {
            if (c != ' ') {
                no_spaces += c;
            }
        }

             if (no_spaces.length() <= max_width) {
            return no_spaces;
        }

        // If still too long, truncate with ellipsis
        if (max_width > 3) {
            return no_spaces.substr(0, max_width - 3) + "...";
        }

        return no_spaces.substr(0, max_width);
}

std::string WaveAnimator::getStyledChar(char c, int style, bool capitalize) {
    char displayChar = capitalize ? std::toupper(c) : c;
    switch(style) {
        case 0: return "\033[32m" + std::string(1, displayChar) + "\033[0m";
        case 1: return "\033[32m" + std::string(1,displayChar) + "\033[0m";
        case 2: return "\033[1m\033[38;5;208m" + std::string(1, displayChar) + "\033[0m"; // Bold + Orange
        default: return "\033[32m" + std::string(1, displayChar) + "\033[0m";
    }
}

void WaveAnimator::waveAnimation(const std::string& text, int cycles) {
        std::cout << "\033[?25l"; // Hide cursor

        // Calculate available width
        int available_width = terminal_width_ - 15;
        if (available_width < 20) available_width = 20;

        // Compress text to fit on one line
        std::string display_text = compressTextToFit(text, available_width);
        int line_length = display_text.length();

        for (int cycle = 0; cycle < cycles; cycle++) {
            for (int pos = 0; pos < line_length + 5; pos++) {
                std::cout << "\r";

                // Apply wave effect
                for (int i = 0; i < line_length; i++) {
                    int distance = std::abs(i - pos);
                    bool shouldCapitalize = (distance == 0);

                    if (distance == 0) {
                        std::cout << getStyledChar(display_text[i], 2, shouldCapitalize);
                    } else if (distance == 1) {
                        std::cout << getStyledChar(display_text[i], 1, shouldCapitalize);
                    } else {
                        std::cout << getStyledChar(display_text[i], 0, shouldCapitalize);
                    }
                }

                                // Rotating spinner
                std::string spinner = "|/-\\";
                std::cout << "\033[36m " << spinner[pos % spinner.length()] << "\033[0m";
                std::cout.flush();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Final state
        std::cout << "\033[32m" << "\r" << display_text << " |\033[0m" << std::endl;
        std::cout << "\033[?25h"; // Show cursor
    }

