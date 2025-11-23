#include "animator.h"

void ConsoleAnimator::printEnlargedChar(char c, bool isLast) {
    const std::string bold = "\033[1m";
    const std::string normal = "\033[0m";
    std::cout << bold << c << normal;
    if (!isLast) {
        std::cout << " ";
    }
}

void ConsoleAnimator::printNormalChar(char c, bool isLast) {
    const std::string normal = "\033[0m";
    std::cout << normal << c;
    if (!isLast) {
        std::cout << " ";
    }
}

void ConsoleAnimator::animateText(const std::string& text, int durationMs) {
    hideCursor();

    int totalChars = text.length();
    int frameDelay = 100;
    int totalFrames = durationMs / frameDelay;
    int charsPerFrame = std::max(1, totalFrames / totalChars);

    // Initial display
    clearLine();
    for (int i = 0; i < totalChars; i ++) {
        printNormalChar(text[i], i == totalChars - 1);
    }

    std::cout.flush();

    // Animation loop
    for (int frame = 0; frame <totalFrames; frame++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(frameDelay));
        clearLine();

        for (int i = 0; i < totalChars; i++) {
            int charFrame = (frame - i * charsPerFrame) % (charsPerFrame * 3);

            if (charFrame >= 0 && charFrame < charsPerFrame) {
                printEnlargedChar(text[i], i == totalChars - 1);
            } else {
                printNormalChar(text[i], i == totalChars - 1);
            }
        }

        std::string spinner = "|/-\\";
        std::cout << " " << spinner[frame % spinner.length()];
    }

    // Final state
    clearLine();
    for (int i = 0; i < totalChars; i++) {
        printNormalChar(text[i], i == totalChars - 1);
    }
    std::cout << " |" << std::endl;
    showCursor();
}


std::string WaveAnimator::getStyledChar(char c, int style, bool capitalize) {
    char displayChar = capitalize ? std::toupper(c) : c;
    switch(style) {
        case 0: return std::string(1, displayChar);
        case 1: return "\033[1m" + std::string(1,displayChar) + "\033[0m";
        case 2: return "\033[1m\033[38;5;208m" + std::string(1, displayChar) + "\033[0m"; // Bold + Orange
        default: return std::string(1, displayChar);
    }
}

void WaveAnimator::waveAnimation(const std::string& text, int cycles) {
    std::cout << "\033[?25l"; // Hide cursor

    for (int cycle = 0; cycle < cycles; cycle++) {
        for (int pos = 0; pos < text.length() + 5; pos++) {
            std::cout << "\r";

            for (int i = 0;  i < text.length(); i++) {
                int distance = std::abs(i - pos);
                bool shouldCapitalize = (distance == 0);

                    if (distance == 0) {
                        std::cout << getStyledChar(text[i], 2, shouldCapitalize); // Peak of wave
                    } else if (distance == 1) {
                        std::cout << getStyledChar(text[i], 1, shouldCapitalize); // Approaching peak
                    } else {
                        std::cout << getStyledChar(text[i], 0, shouldCapitalize); // Normal
                    }

                    if (i < text.length() - 1) {
                        std::cout << " ";
                    }
                }

                std::string spinner = "|/-\\";
                std::cout << " " << spinner[pos % spinner.length()];
                                std::cout.flush();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Final state - all normal, no capitalization
        std::cout << "\r";
        for (int i = 0; i < text.length(); i++) {
            std::cout << text[i];
            if (i < text.length() - 1) {
                std::cout << " ";
            }
        }
        std::cout << " |" << std::endl;
        std::cout << "\033[?25h"; // Show cursor
    }

