#ifndef ANIMATOR_H
#define ANIMATOR_H

#include <string>
#include <iostream>
#include <thread>
#include <chrono>

class ConsoleAnimator {
    private:
        void hideCursor() {
            std::cout << "\033[?25l";
        }

        void showCursor() {
            std::cout << "\033[?25h";
        }

        void clearLine() {
            std::cout << "\033[2k\r";
        }

    public:
        ConsoleAnimator() = default;
        ~ConsoleAnimator() {
            showCursor();
        }

        void printEnlargedChar(char c, bool isLast = false);
        void printNormalChar(char c, bool isLast = false);
        void animateText(const std::string& text, int durationMs = 3000);
};

class WaveAnimator {
    private:
        std::string getStyledChar(char c, int style, bool capitalize);
    public:
        void waveAnimation(const std::string& text, int cycles = 2);
};

#endif
