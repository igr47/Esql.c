#ifndef ANIMATOR_H
#define ANIMATOR_H

#include <string>
#include <iostream>
#include <thread>
#include <chrono>

class ConsoleAnimator {
    private:
        int terminal_width_;
        void hideCursor() {
            std::cout << "\033[?25l";
        }

        void showCursor() {
            std::cout << "\033[?25h";
        }

        void clearLine() {
            std::cout << "\033[2k\r";
        }
        std::string compressTextToFit(const std::string& text, int max_width);

    public:
        ConsoleAnimator(int terminal_width = 80) : terminal_width_(terminal_width) {}
        ~ConsoleAnimator() {
            showCursor();
        }

        void printEnlargedChar(char c, bool isLast = false);
        void printNormalChar(char c, bool isLast = false);
        void animateText(const std::string& text, int durationMs = 3000);
};

class WaveAnimator {
    private:
        int terminal_width_;
        std::string getStyledChar(char c, int style, bool capitalize);
        std::string compressTextToFit(const std::string& text, int max_width);
    public:
        WaveAnimator(int terminal_width = 80) : terminal_width_(terminal_width) {}
        void waveAnimation(const std::string& text, int cycles = 2);
};

#endif
